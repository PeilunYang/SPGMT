import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
import time
import datetime
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


class TimeEmbedding(nn.Module):
    def __init__(self, date2vec_size, device):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.date2vec_size = date2vec_size

    def forward(self, time_seqs):
        """
        padding and timestamp series embedding
        :param time_seqs: list [batch,timestamp_seq]
        :return: packed_input
        """
        batch_size = len(time_seqs)
        seq_lengths = list(map(len, time_seqs))

        for time_one in time_seqs:
            time_one += [[0 for i in range(self.date2vec_size)]]*(max(seq_lengths)-len(time_one))

        # vec_time_seqs = self.d2vec(time_seqs).to(self.device)

        # prepare sequence tensor
        embedded_seq_tensor = torch.zeros((batch_size, max(seq_lengths), self.date2vec_size), dtype=torch.float32)

        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        # time_seqs = torch.tensor(time_seqs).to(self.device)
        vec_time_seqs = torch.tensor(time_seqs).to(self.device)

        # get embedding for trajectory embeddings
        for idx, (seq, seqlen) in enumerate(zip(vec_time_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = seq[:seqlen]

        # move to cuda device
        seq_lengths = seq_lengths.cpu()
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)

        # packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths, batch_first=True,enforce_sorted=False)
        return embedded_seq_tensor


class Co_Att(nn.Module):
    def __init__(self, dim):
        super(Co_Att, self).__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.temperature = dim ** 0.5
        self.FFN = nn.Sequential(
            nn.Linear(dim, int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5), dim),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, seq_s, seq_t):
        h = torch.stack([seq_s, seq_t], 2)  # [n, 2, dim]
        # print('shape of h is: ', h.shape)
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # print('shape of attn is: ', attn.shape)
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)

        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)

        att_s = attn_o[:, :, 0, :]
        att_t = attn_o[:, :, 1, :]

        return att_s, att_t


class ST_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_rate, device):
        super(ST_LSTM, self).__init__()
        self.device = device
        self.bi_lstm = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout_rate,
                               bidirectional=True)
        # self-attention weights
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def getMask(self, seq_lengths):
        """
        create mask based on the sentence lengths
        :param seq_lengths: sequence length after `pad_packed_sequence`
        :return: mask (batch_size, max_seq_len)
        """
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0

        return mask

    def forward(self, packed_input):
        # output features (h_t) from the last layer of the LSTM, for each t
        # (batch_size, seq_len, 2 * num_hiddens)
        packed_output, _ = self.bi_lstm(packed_input)  # output, (h, c)
        outputs, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # get sequence mask
        mask = self.getMask(seq_lengths)

        # Attention...
        # (batch_size, seq_len, 2 * num_hiddens)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        # (batch_size, seq_len)
        att = torch.matmul(u, self.u_omega).squeeze()

        # add mask
        att = att.masked_fill(mask == 0, -1e10)

        # (batch_size, seq_len,1)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        # normalization attention weight
        # (batch_size, seq_len, 2 * num_hiddens)
        scored_outputs = outputs * att_score

        # weighted sum as output
        # (batch_size, 2 * num_hiddens)
        out = torch.sum(scored_outputs, dim=1)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, device, dropout: float = 0.1, max_len: int = 3700):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        #print(x.device)
        #print(torch.tensor(self.pe[:x.size(0)]).to(self.device).device)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float, device):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, device, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.decoder(output)
        return output


class PosKnnGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, usePE, useSI, dataset):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.usePE = usePE
        self.useSI = useSI
        print('usePE: ' + str(self.usePE) + ', useSI: ' + str(self.useSI))
        self.dataset = dataset
        if dataset == 'beijing':
            self.nodeLin = torch.nn.Linear(in_channels + 98, in_channels, bias=False)
        elif dataset == 'tdrive':
            self.nodeLin = torch.nn.Linear(in_channels + 112, in_channels, bias=False)
        elif dataset == 'porto':
            self.nodeLin = torch.nn.Linear(in_channels + 68, in_channels, bias=False)

        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.nodeLin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, input_edge_index, input_edge_attr, d2an, firstLayer):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_attr = add_self_loops(input_edge_index, input_edge_attr, num_nodes=x.size(0), fill_value=1.0)

        # Step 2: Linearly transform node feature matrix.
        if firstLayer and self.usePE:
            combined_input = torch.cat((x, d2an), dim=1)
            x = self.nodeLin(combined_input)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        edge_inv_sqrt = edge_attr.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_inv_sqrt[edge_inv_sqrt > 1.0] = 1.0
        edge_norm = edge_inv_sqrt

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=[deg_norm, edge_norm])

        return out

    def gcn_forward(self, x, input_edge_index, input_edge_attr, d2an, firstLayer):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_attr = add_self_loops(input_edge_index, input_edge_attr, num_nodes=x.size(0), fill_value=1.0)

        # Step 2: Linearly transform node feature matrix.
        if firstLayer and self.usePE:
            combined_input = torch.cat((x, d2an), dim=1)
            x = self.nodeLin(combined_input)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        edge_inv_sqrt = edge_attr.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_inv_sqrt[edge_inv_sqrt > 1.0] = 1.0
        edge_norm = edge_inv_sqrt

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=deg_norm)

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        if self.useSI:
            return norm[0].view(-1, 1) * (self.lin1(x_j)) + norm[1].view(-1, 1) * (self.lin2(x_j))
        else:
            return norm.view(-1, 1) * (self.lin1(x_j))


class KnnGNN(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, useSI, dataset):
        super(KnnGNN, self).__init__()
        self.usePE = usePE
        self.useSI = useSI
        self.dataset = dataset
        self.posconv1 = PosKnnGNNLayer(encoding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.posconv2 = PosKnnGNNLayer(encoding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.posconv3 = PosKnnGNNLayer(embedding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.posconv4 = PosKnnGNNLayer(embedding_size, embedding_size, self.usePE, self.useSI, self.dataset)

    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index_l0, edge_weight_l0 = data[0].x, data[0].edge_index, data[0].edge_attr
        _, edge_index_l1, edge_weight_l1 = data[1].x, data[1].edge_index, data[1].edge_attr
        _, edge_index_l2, edge_weight_l2 = data[2].x, data[2].edge_index, data[2].edge_attr

        x0 = F.relu(self.posconv1(x, edge_index_l0, edge_weight_l0, d2an, True))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x1 = F.relu(self.posconv2(x, edge_index_l1, edge_weight_l1, d2an, True))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x = 0.75 * x0 + 0.25 * x1

        x0 = F.relu(self.posconv3(x, edge_index_l0, edge_weight_l0, d2an, False))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x1 = F.relu(self.posconv4(x, edge_index_l1, edge_weight_l1, d2an, False))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x = 0.75 * x0 + 0.25 * x1

        return x

    def noSI_forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index_l0, edge_weight_l0 = data.x, data.edge_index, data.edge_attr

        x0 = F.relu(self.posconv1.gcn_forward(x, edge_index_l0, edge_weight_l0, d2an, True))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x = x0

        x0 = F.relu(self.posconv3.gcn_forward(x, edge_index_l0, edge_weight_l0, d2an, False))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x = x0

        return x


class GraphTrajSTEncoder(nn.Module):
    def __init__(self, feature_size, embedding_size, date2vec_size, hidden_size, num_layers, dropout_rate, concat, device, usePE, dataset):
        super(GraphTrajSTEncoder, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.usePE = usePE
        self.dataset = dataset
        self.graph_embedding = KnnGNN(feature_size, embedding_size, self.usePE, self.dataset)
        self.time_embedding = TimeEmbedding(date2vec_size, device)
        self.co_attention = Co_Att(date2vec_size).to(device)
        self.encoder_ST = ST_LSTM(embedding_size+date2vec_size, hidden_size, num_layers, dropout_rate, device)

    def obtain_trm_src_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l <= max_len:
                mask[i, :l] = 0
                # mask[i, :, l:] = 0

        return mask.bool()

    def forward(self, network_data, traj_seqs, time_seqs):
        batch_size = len(traj_seqs)
        seq_lengths = list(map(len, traj_seqs))

        for traj_one in traj_seqs:
            traj_one += [0] * (max(seq_lengths) - len(traj_one))

        # prepare sequence tensor
        embedded_seq_tensor = torch.zeros((batch_size, max(seq_lengths), self.embedding_size), dtype=torch.float32)  # .to('cpu')
        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        traj_seqs = torch.tensor(traj_seqs).to(self.device)

        # get node embeddings from gcn
        # (num_nodes, embedding_size)
        graph_node_embeddings = self.graph_embedding(network_data)

        for idx, (seq, seqlen) in enumerate(zip(traj_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = graph_node_embeddings.index_select(0, seq[:seqlen])

        spa_input = embedded_seq_tensor.to(self.device)
        time_input = self.time_embedding(time_seqs)
        att_s, att_t = self.co_attention(spa_input, time_input)
        st_input = torch.cat((att_s, att_t), dim=2)
        seq_lengths = seq_lengths.to('cpu')
        packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)

        out = self.encoder_ST(packed_input)

        return out


class GraphTrajSimEncoder(nn.Module):
    def __init__(self, feature_size, embedding_size, hidden_size, num_layers, dropout_rate, concat, device, usePE, useSI, dataset):
        super(GraphTrajSimEncoder, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.usePE = usePE
        self.useSI = useSI
        self.dataset = dataset
        self.graph_embedding = KnnGNN(feature_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.trm_encoder = TransformerModel(hidden_size, embedding_size, 4, 512, 1, 0.3, device)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def obtain_trm_src_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l <= max_len:
                mask[i, :l] = 0

        return mask.bool()

    def forward(self, network_data, traj_seqs):
        batch_size = len(traj_seqs)
        seq_lengths = list(map(len, traj_seqs))

        for traj_one in traj_seqs:
            traj_one += [0] * (max(seq_lengths) - len(traj_one))

        # prepare sequence tensor
        embedded_seq_tensor = torch.zeros((batch_size, max(seq_lengths), self.embedding_size), dtype=torch.float32)
        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        traj_seqs = torch.tensor(traj_seqs).to(self.device)

        # get node embeddings from gcn, (num_nodes, embedding_size)
        if self.useSI:
            graph_node_embeddings = self.graph_embedding(network_data)
        else:
            graph_node_embeddings = self.graph_embedding.noSI_forward(network_data)

        for idx, (seq, seqlen) in enumerate(zip(traj_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = graph_node_embeddings.index_select(0, seq[:seqlen])

        embedded_seq_tensor = embedded_seq_tensor.to(self.device)

        # (batch_size, max_len, embedding_size), trm
        trm_encoder_src_mask = self.obtain_trm_src_mask(seq_lengths)
        trm_outputs = self.trm_encoder(embedded_seq_tensor.transpose(1, 0), trm_encoder_src_mask).transpose(1, 0)
        u = torch.tanh(torch.matmul(trm_outputs, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att = att.masked_fill(trm_encoder_src_mask == True, -1e10)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = trm_outputs * att_score
        out = torch.sum(scored_outputs, dim=1)

        return out
