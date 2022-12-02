import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import random
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import spatial_similarity_computation as spatial_com
import pickle
import yaml
import os

random.seed(1933)
np.random.seed(1933)
config = yaml.safe_load(open('config.yaml'))


def load_netowrk(dataset):
    edge_path = str(config["edge_file"])
    node_embedding_path = "./data/" + dataset + "/node_features.npy"

    node_embeddings = np.load(node_embedding_path)
    df_dege = pd.read_csv(edge_path, sep=',')

    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    edge_attr = df_dege["length"].to_numpy()
    if str(config["dataset"]) == "beijing" or "porto":
        edge_attr = edge_attr / 100.0

    edge_index = torch.LongTensor(edge_index).t().contiguous()
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    print("node embeddings shape: ", node_embeddings.shape)
    print("edge_index shap: ", edge_index.shape)
    print("edge_attr shape: ", edge_attr.shape)

    road_network = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)

    return road_network


def load_neighbor(dataset):
    k5_neighbor = np.load('./dataset/' + dataset + '/k5_neighbor.npy')
    k5_distance = np.load('./dataset/' + dataset + '/k5_distance.npy')
    k10_neighbor = np.load('./dataset/' + dataset + '/k10_neighbor.npy')
    k10_distance = np.load('./dataset/' + dataset + '/k10_distance.npy')
    k15_neighbor = np.load('./dataset/' + dataset + '/k15_neighbor.npy')
    k15_distance = np.load('./dataset/' + dataset + '/k15_distance.npy')
    node_embedding_path = "./data/" + dataset + "/node_features.npy"
    degree_encodings = np.load(node_embedding_path)

    edge_index_l0 = torch.LongTensor(k5_neighbor).t().contiguous()
    edge_attr_l0 = torch.tensor(k5_distance, dtype=torch.float)
    edge_index_l1 = torch.LongTensor(k10_neighbor).t().contiguous()
    edge_attr_l1 = torch.tensor(k10_distance, dtype=torch.float)
    edge_index_l2 = torch.LongTensor(k15_neighbor).t().contiguous()
    edge_attr_l2 = torch.tensor(k15_distance, dtype=torch.float)
    node_embeddings = torch.tensor(degree_encodings, dtype=torch.float)

    print("node embeddings shape: ", node_embeddings.shape)
    print("edge_index_l0 shape: ", edge_index_l0.shape)
    print("edge_attr_l0 shape: ", edge_attr_l0.shape)
    print("edge_index_l1 shape: ", edge_index_l1.shape)
    print("edge_attr_l1 shape: ", edge_attr_l1.shape)
    print("edge_index_l2 shape: ", edge_index_l2.shape)
    print("edge_attr_l2 shape: ", edge_attr_l2.shape)

    road_network_l0 = Data(x=node_embeddings, edge_index=edge_index_l0, edge_attr=edge_attr_l0)
    road_network_l1 = Data(x=[], edge_index=edge_index_l1, edge_attr=edge_attr_l1)
    road_network_l2 = Data(x=[], edge_index=edge_index_l2, edge_attr=edge_attr_l2)

    return [road_network_l0, road_network_l1, road_network_l2]


def sample_region_node(dataset):
    if dataset == 'beijing':
        node_file = str(config["node_file"])
        df_node = pd.read_csv(node_file, sep=',')
        all_node, all_lng, all_lat = df_node.node, df_node.lng, df_node.lat
        all_node = np.array(all_node)
        all_lng = np.array(all_lng)
        all_lat = np.array(all_lat)
        point_dis = np.load('./ground_truth/' + dataset + '/Point_dis_matrix.npy')
        cnt_matrix = np.zeros((7, 8))
        region_node = [[[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []],
                       [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []],
                       [[], [], [], [], [], [], [], []]]
        for i in range(112557):
            if 115.5 < all_lng[i] < 117.5 and 39 < all_lat[i] < 40.75:
                node_id = all_node[i]
                node_lng = int((all_lng[i] - 115.5) / 0.25)
                node_lat = int((all_lat[i] - 39) / 0.25)
                region_node[node_lat][node_lng].append(node_id)
                cnt_matrix[node_lat][node_lng] += 1

        selected_node_set = []
        for i in range(7):
            for j in range(8):
                node_list = region_node[i][j]
                if len(node_list) == 1:
                    selected_node_set.append(node_list[0])
                elif len(node_list) == 2:
                    selected_node_set.append(node_list[0])
                    selected_node_set.append(node_list[1])
                elif len(node_list) > 2:
                    node_ids = list(np.random.choice(len(node_list), 2, replace=False))
                    selected_node_set.append(node_ids[0])
                    selected_node_set.append(node_ids[1])

        all_distance_to_node = []
        for item in selected_node_set:
            tmp = point_dis[:112557, item]
            ids = np.where(tmp != -1)
            for idx in ids:
                tmp[idx] = np.exp(-(tmp[idx]/100.0))
            all_distance_to_node.append(tmp)

        all_distance_to_node = np.array(all_distance_to_node).T
        distance_to_anchor_node = torch.tensor(all_distance_to_node, dtype=torch.float)
        print("distance to anchor nodes shape: ", distance_to_anchor_node.shape)

        return distance_to_anchor_node

    elif dataset == 'tdrive':
        node_file = './data/tdrive/road/node.csv'
        df_node = pd.read_csv(node_file, sep=',')
        all_node, all_lng, all_lat = df_node.node, df_node.lng, df_node.lat
        all_node = np.array(all_node)
        all_lng = np.array(all_lng)
        all_lat = np.array(all_lat)
        point_dis = np.load('/home/peyang/Data/ST2Vec/ST2Vec/ground_truth/' + dataset + '/Point_dis_matrix.npy')

        cnt_matrix = np.zeros((8, 7))
        region_node = [[[], [], [], [], [], [], []], [[], [], [], [], [], [], []], [[], [], [], [], [], [], []],
                       [[], [], [], [], [], [], []], [[], [], [], [], [], [], []], [[], [], [], [], [], [], []],
                       [[], [], [], [], [], [], []], [[], [], [], [], [], [], []]]
        for i in range(74671):
            if 116.1 < all_lng[i] < 116.8 and 39.5 < all_lat[i] < 40.3:
                node_id = all_node[i]
                node_lng = int((all_lng[i] - 116.1) / 0.1)
                node_lat = int((all_lat[i] - 39.5) / 0.1)
                region_node[node_lat][node_lng].append(node_id)
                cnt_matrix[node_lat][node_lng] += 1

        selected_node_set = []
        for i in range(8):
            for j in range(7):
                node_list = region_node[i][j]
                if len(node_list) == 1:
                    selected_node_set.append(node_list[0])
                elif len(node_list) == 2:
                    selected_node_set.append(node_list[0])
                    selected_node_set.append(node_list[1])
                elif len(node_list) > 2:
                    node_ids = list(np.random.choice(len(node_list), 2, replace=False))
                    selected_node_set.append(node_ids[0])
                    selected_node_set.append(node_ids[1])

        all_distance_to_node = []
        for item in selected_node_set:
            tmp = point_dis[:74671, item]
            ids = np.where(tmp != -1)
            for idx in ids:
                tmp[idx] = np.exp(-(tmp[idx] / 100.0))
            all_distance_to_node.append(tmp)

        all_distance_to_node = np.array(all_distance_to_node).T
        distance_to_anchor_node = torch.tensor(all_distance_to_node, dtype=torch.float)
        # for i in range(d2an_len):
        #     distance_to_anchor_node[i][[distance_to_anchor_node[i] != -1.0]] = torch.exp(-(distance_to_anchor_node[i]/100))
        # distance_to_anchor_node[distance_to_anchor_node != -1.0] = torch.exp(-(distance_to_anchor_node/100))
        print("distance to anchor nodes shape: ", distance_to_anchor_node.shape)

        return distance_to_anchor_node

    elif dataset == 'porto':
        node_file = str(config["node_file"])
        df_node = pd.read_csv(node_file, sep=',')
        all_node, all_lng, all_lat = df_node.node, df_node.lng, df_node.lat
        all_node = np.array(all_node)
        all_lng = np.array(all_lng)
        all_lat = np.array(all_lat)
        point_dis = np.load('./ground_truth/' + dataset + '/Point_dis_matrix.npy')
        cnt_matrix = np.zeros((6, 6))
        region_node = [[[], [], [], [], [], []], [[], [], [], [], [], []], [[], [], [], [], [], []],
                       [[], [], [], [], [], [], []], [[], [], [], [], [], []], [[], [], [], [], [], []]]
        for i in range(128466):
            if -8.8 < all_lng[i] < -8.2 and 40.9 < all_lat[i] < 41.5:
                node_id = all_node[i]
                node_lng = int((all_lng[i] + 8.8) / 0.1)
                node_lat = int((all_lat[i] - 40.9) / 0.1)
                region_node[node_lat][node_lng].append(node_id)
                cnt_matrix[node_lat][node_lng] += 1

        selected_node_set = []
        for i in range(6):
            for j in range(6):
                node_list = region_node[i][j]
                if len(node_list) == 1:
                    if np.sum(point_dis[:128466, node_list[0]]) != -128465:
                        selected_node_set.append(node_list[0])
                elif len(node_list) == 2:
                    if np.sum(point_dis[:128466, node_list[0]]) != -128465:
                        selected_node_set.append(node_list[0])
                    if np.sum(point_dis[:128466, node_list[1]]) != -128465:
                        selected_node_set.append(node_list[1])
                elif len(node_list) > 2:
                    node_ids = list(np.random.choice(len(node_list), 2, replace=False))
                    flag = True
                    while flag:
                        if np.sum(point_dis[:128466, node_ids[0]]) != -128465 and np.sum(
                                point_dis[:128466, node_ids[1]]) != -128465:
                            selected_node_set.append(node_ids[0])
                            selected_node_set.append(node_ids[1])
                            flag = False
                        else:
                            node_ids = list(np.random.choice(len(node_list), 2, replace=False))

        all_distance_to_node = []
        for item in selected_node_set:
            tmp = point_dis[:128466, item]
            ids = np.where(tmp != -1)
            for idx in ids:
                tmp[idx] = np.exp(-(tmp[idx] / 100.0))
            all_distance_to_node.append(tmp)

        all_distance_to_node = np.array(all_distance_to_node).T
        distance_to_anchor_node = torch.tensor(all_distance_to_node, dtype=torch.float)
        print("distance to anchor nodes shape: ", distance_to_anchor_node.shape)

        return distance_to_anchor_node


class DataLoader():
    def __init__(self):
        self.kseg = config["kseg"]
        self.train_set = config["train_set_size"]
        self.vali_set = config["vali_set_size"]
        self.test_set = config["test_set_size"]

    def load(self, load_part):
        node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)

        train_set = self.train_set
        vali_set = self.vali_set
        test_set = self.test_set

        if load_part=='train':
            return node_list_int[:train_set]
        if load_part=='vali':
            return node_list_int[train_set:vali_set]
        if load_part=='test':
            return node_list_int[vali_set:test_set]
        if load_part=='long':
            tmp_node_list = node_list_int[vali_set:test_set]
            long_ids = []
            long_trajs = []
            long_thres = 150

            for idx, item in enumerate(tmp_node_list):
                if len(item) >= long_thres:
                    long_ids.append(idx)
                    long_trajs.append(item)

            print('The number of long trajectories is ' + str(len(long_ids)))

            return long_ids, tmp_node_list
        if load_part=='scale':
            node_list_int = np.load('./data/beijing/st_traj/30k_node_list.npy', allow_pickle=True)
            test_num = 300000
            return node_list_int[:test_num]

    def ksegment_ST(self):
        kseg_coor_trajs = np.load(str(config["shuffle_kseg_file"]), allow_pickle=True)[:self.train_set]

        max_lat = -1000
        max_lon = -1000
        for traj in kseg_coor_trajs:
            for t in traj:
                if max_lat<t[0]:
                    max_lat = t[0]
                if max_lon<t[1]:
                    max_lon = t[1]
        print(max_lat, max_lon)
        kseg_coor_trajs = kseg_coor_trajs/[max_lat,max_lon]
        kseg_coor_trajs = kseg_coor_trajs.reshape(-1,self.kseg*2)

        return kseg_coor_trajs

    def get_triplets(self):
        train_node_list = self.load(load_part='train')

        sample_train2D = self.ksegment_ST()

        ball_tree = BallTree(sample_train2D)

        anchor_index = list(range(len(train_node_list)))
        random.shuffle(anchor_index)

        apn_node_triplets = []
        for j in range(1,1001):
            for i in anchor_index:
                dist, index = ball_tree.query([sample_train2D[i]], j+1)  # k nearest neighbors
                p_index = list(index[0])
                p_index = p_index[-1]

                p_sample = train_node_list[p_index]  # positive sample
                n_index = random.randint(0, len(train_node_list)-1)
                while n_index == i:
                    n_index = random.randint(0, len(train_node_list)-1)
                n_sample = train_node_list[n_index]  # negative sample
                a_sample = train_node_list[i]  # anchor sample

                ok = True
                if str(config["distance_type"]) == "TP":
                    if spatial_com.TP_dis(a_sample,p_sample)==-1 or spatial_com.TP_dis(a_sample,n_sample)==-1:
                        ok = False
                elif str(config["distance_type"]) == "DITA":
                    if spatial_com.DITA_dis(a_sample,p_sample)==-1 or spatial_com.DITA_dis(a_sample,n_sample)==-1:
                        ok = False
                elif str(config["distance_type"]) == "discret_frechet":
                    if spatial_com.frechet_dis(a_sample,p_sample)==-1 or spatial_com.frechet_dis(a_sample,n_sample)==-1:
                        ok = False
                elif str(config["distance_type"]) == "LCRS":
                    if spatial_com.LCRS_dis(a_sample, p_sample) == spatial_com.longest_traj_len*2 or spatial_com.LCRS_dis(a_sample, n_sample) == spatial_com.longest_traj_len*2:
                        ok = False
                elif str(config["distance_type"]) == "NetERP":
                    if spatial_com.NetERP_dis(a_sample,p_sample)==-1 or spatial_com.NetERP_dis(a_sample,n_sample)==-1:
                        ok = False

                if ok:
                    apn_node_triplets.append([a_sample, p_sample, n_sample])

                if len(apn_node_triplets)==len(train_node_list)*2:
                    break
            if len(apn_node_triplets) == len(train_node_list)*2:
                break
        print("complete: sample")
        print("complete: sample")
        print(apn_node_triplets[0])
        p = './data/{}/triplet/{}/'.format(str(config["dataset"]), str(config["distance_type"]))
        if not os.path.exists(p):
            os.makedirs(p)
        pickle.dump(apn_node_triplets,open(str(config["spatial_path_node_triplets"]),'wb'))

    def return_triplets_num(self):
        apn_node_triplets = pickle.load(open(str(config["spatial_path_node_triplets"]), 'rb'))
        return len(apn_node_triplets)

def triplet_groud_truth():
    apn_node_triplets = pickle.load(open(str(config["spatial_path_node_triplets"]),'rb'))
    com_max_s = []

    for i in range(len(apn_node_triplets)):
        if str(config["distance_type"]) == "TP":
            ap_s = spatial_com.TP_dis(apn_node_triplets[i][0],apn_node_triplets[i][1])
            an_s = spatial_com.TP_dis(apn_node_triplets[i][0],apn_node_triplets[i][2])
            com_max_s.append([ap_s, an_s])
        elif str(config["distance_type"]) == "DITA":
            ap_s = spatial_com.DITA_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
            an_s = spatial_com.DITA_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
            com_max_s.append([ap_s, an_s])
        elif str(config["distance_type"]) == "discret_frechet":
            ap_s = spatial_com.frechet_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
            an_s = spatial_com.frechet_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
            com_max_s.append([ap_s, an_s])
        elif str(config["distance_type"]) == "LCRS":
            ap_s = spatial_com.LCRS_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
            an_s = spatial_com.LCRS_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
            com_max_s.append([ap_s, an_s])
        elif str(config["distance_type"]) == "NetERP":
            ap_s = spatial_com.NetERP_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
            an_s = spatial_com.NetERP_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
            com_max_s.append([ap_s, an_s])

    com_max_s = np.array(com_max_s)

    if str(config["dataset"]) == "tdrive" or "beijing" or "porto":
        if str(config["distance_type"]) == "TP":
            coe = 8
        elif str(config["distance_type"]) == "DITA":
            coe = 32
        elif str(config["distance_type"]) == "LCRS":
            coe = 4
        elif str(config["distance_type"]) == "NetERP":
            coe = 8
        elif str(config["distance_type"]) == "discret_frechet":
            coe = 8
    if str(config["dataset"]) == "rome":
        if str(config["distance_type"]) == "TP":
            coe = 8
        elif str(config["distance_type"]) == "DITA":
            coe = 16
        elif str(config["distance_type"]) == "LCRS":
            coe = 2
        elif str(config["distance_type"]) == "NetERP":
            coe = 8

    com_max_s = com_max_s/ np.max(com_max_s) * coe

    train_triplets_dis = (com_max_s+com_max_s)/2

    np.save(str(config["spatial_path_triplets_truth"]), train_triplets_dis)
    print("complete: triplet ground truth")
    print(train_triplets_dis[0])


def test_merge_st_dis(valiortest = None):
    s = np.load('./ground_truth/{}/{}/{}_spatial_distance_{}.npy'.format(str(config["dataset"]), str(config["distance_type"]), valiortest, str(config["dataset_size"])))
    print(s.shape)

    unreach = {}
    for i, dis in enumerate(s):
        tmp = []
        for j, und in enumerate(dis):
            if und == -1:
                tmp.append(j)
        if len(tmp) > 0:
            unreach[i] = tmp

    s = s / np.max(s)

    for i in unreach.keys():
        s[i][unreach[i]]=-1

    if valiortest == 'vali':
        np.save(str(config["spatial_path_vali_truth"]), s)
    else:
        np.save(str(config["spatial_path_test_truth"]), s)

    print("complete: merge_spa_distance")

class batch_list():
    def __init__(self, batch_size):
        self.apn_node_triplets = np.array(pickle.load(open(str(config["spatial_path_node_triplets"]), 'rb')))
        self.batch_size = batch_size
        self.start = len(self.apn_node_triplets)

    def getbatch_one(self, shf):
        # batch random
        if shf:
            index = list(range(len(self.apn_node_triplets)))
            random.shuffle(index)
            batch_index = random.sample(index, self.batch_size)

        # batch reverse
        if not shf:
            if self.start - self.batch_size < 0:
                self.start = len(self.apn_node_triplets)
            batch_index = list(range(self.start - self.batch_size, self.start))
            self.start -= self.batch_size

        node_list = self.apn_node_triplets[batch_index]

        a_node_batch = []
        a_time_batch = []
        p_node_batch = []
        p_time_batch = []
        n_node_batch = []
        n_time_batch = []
        for tri1 in node_list:
            a_node_batch.append(tri1[0])
            p_node_batch.append(tri1[1])
            n_node_batch.append(tri1[2])

        return a_node_batch, p_node_batch, n_node_batch, batch_index


if __name__ == "__main__":
    data = DataLoader()
    data.get_triplets()
    triplet_groud_truth()
    test_merge_st_dis(valiortest='vali')
    test_merge_st_dis(valiortest='test')


