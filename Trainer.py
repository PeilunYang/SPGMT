from model_network import GraphTrajSTEncoder, GraphTrajSimEncoder
import yaml
import torch
# import data_utils
import spatial_data_utils
from lossfun import STLossFun, SpaLossFun
import test_method
import time
import random
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
import logging


class GTrajST_Trainer(object):
    def __init__(self):
        config = yaml.safe_load(open('config.yaml'))

        self.feature_size = config["feature_size"]
        self.embedding_size = config["embedding_size"]
        self.date2vec_size = config["date2vec_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.concat = config["concat"]
        self.device = "cuda:" + str(config["cuda"])
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]

        self.train_batch = config["gtraj"]["train_batch"]
        self.test_batch = config["gtraj"]["test_batch"]
        self.usePE = config["gtraj"]["usePE"]
        self.traj_file = str(config["traj_file"])
        self.time_file = str(config["time_file"])

        self.dataset = str(config["dataset"])
        self.distance_type = str(config["distance_type"])
        self.early_stop = config["early_stop"]
        print('SPGMT on ' + self.dataset + ' with ' + self.distance_type + ' (ST)')

    def ST_eval(self, load_model=None):
        net = GraphTrajSTEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device,
                               usePE=self.usePE,
                               dataset=self.dataset)

        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)

            dataload = data_utils.DataLoader()
            road_network = spatial_data_utils.load_neighbor(self.dataset)
            distance_to_anchor_node = spatial_data_utils.sample_region_node(self.dataset).to(self.device)
            for item in road_network:
                item = item.to(self.device)

            net.eval()
            with torch.no_grad():
                vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='test')
                embedding_vali = test_method.compute_embedding(road_network=[road_network, distance_to_anchor_node], net=net,
                                                               test_traj=list(vali_node_list),
                                                               test_time=list(vali_d2vec_list),
                                                               test_batch=self.test_batch)
                acc = test_method.test_model(self.dataset, self.distance_type, embedding_vali, isvali=False)
                print(acc)

    def ST_train(self, load_model=None, load_optimizer=None):

        net = GraphTrajSTEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device,
                               usePE=self.usePE,
                               dataset=self.dataset)

        dataload = data_utils.DataLoader()
        # dataload.get_triplets()
        # data_utils.triplet_groud_truth()

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        lossfunction = STLossFun(self.train_batch, self.distance_type)

        net.to(self.device)
        lossfunction.to(self.device)

        road_network = spatial_data_utils.load_neighbor(self.dataset)
        distance_to_anchor_node = spatial_data_utils.sample_region_node(self.dataset).to(self.device)
        for item in road_network:
            item = item.to(self.device)

        bt_num = int(dataload.return_triplets_num() / self.train_batch)

        batch_l = data_utils.batch_list(batch_size=self.train_batch)

        best_epoch = 0
        best_hr10 = 0
        lastepoch = '0'
        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            optimizer.load_state_dict(torch.load(load_optimizer))
            lastepoch = load_model.split('/')[-1].split('_')[3]
            best_epoch = int(lastepoch)

        accumulation_steps = 128 / self.train_batch
        for epoch in range(int(lastepoch), self.epochs):
            net.train()
            accumulation_loss = 0.0
            s1 = time.time()
            for bt in range(bt_num):
                a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index = batch_l.getbatch_one()

                a_embedding = net([road_network, distance_to_anchor_node], a_node_batch, a_time_batch)
                p_embedding = net([road_network, distance_to_anchor_node], p_node_batch, p_time_batch)
                n_embedding = net([road_network, distance_to_anchor_node], n_node_batch, n_time_batch)

                loss = lossfunction(a_embedding, p_embedding, n_embedding, batch_index)
                loss = loss / accumulation_steps
                accumulation_loss += loss

                loss.backward()

                if ((bt + 1) % accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if (bt_num + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            s5 = time.time()
            print('Epoch {}: Training time is {}, loss is {}'.format(epoch, (s5 - s1), accumulation_loss))
            if epoch % 2 == 0:
                net.eval()
                with torch.no_grad():
                    s6 = time.time()
                    vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali')
                    embedding_vali = test_method.compute_embedding(road_network=[road_network, distance_to_anchor_node], net=net,
                                                                   test_traj=list(vali_node_list),
                                                                   test_time=list(vali_d2vec_list),
                                                                   test_batch=self.test_batch)
                    acc = test_method.test_model(self.dataset, self.distance_type, embedding_vali, isvali=True)
                    s7 = time.time()
                    print("test time: ", s7 - s6)
                    print(acc)

                    # save model
                    save_modelname = './model/tdrive_ST/Our_{}/coe128/{}_{}_epoch_{}_HR10_{}_HR50_{}_HR1050_{}_trainLoss_{}.pkl'.format(
                        self.distance_type, self.dataset, self.distance_type, str(epoch), acc[0], acc[1], acc[2], accumulation_loss.item())
                    torch.save(net.state_dict(), save_modelname)

                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                    # if epoch - best_epoch >= self.early_stop:
                    #    break


class GTrajsim_Trainer(object):
    def __init__(self):
        config = yaml.safe_load(open('config.yaml'))

        self.feature_size = config["feature_size"]
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.concat = config["concat"]
        self.device = "cuda:" + str(config["cuda"])
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]

        self.train_batch = config["gtraj"]["train_batch"]
        self.test_batch = config["gtraj"]["test_batch"]
        self.traj_file = str(config["traj_file"])
        self.usePE = config["gtraj"]["usePE"]
        self.useSI = config["gtraj"]["useSI"]

        self.dataset = str(config["dataset"])
        self.distance_type = str(config["distance_type"])
        self.early_stop = config["early_stop"]
        print('SPGMT on ' + self.dataset + ' with ' + self.distance_type)

    def Spa_eval(self, load_model=None):
        net = GraphTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device,
                               usePE=self.usePE,
                               useSI=self.useSI,
                               dataset=self.dataset)

        if load_model != None:
            print(load_model)
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)

            dataload = spatial_data_utils.DataLoader()
            if self.useSI:
                road_network = spatial_data_utils.load_neighbor(self.dataset)
                for item in road_network:
                    item = item.to(self.device)
            else:
                road_network = spatial_data_utils.load_netowrk(self.dataset).to(self.device)
            distance_to_anchor_node = spatial_data_utils.sample_region_node(self.dataset).to(self.device)

            net.eval()
            with torch.no_grad():
                # if 'load_part' == 'test', SPGMT is tested using test data; if 'load_part' == 'long', SPGMT is tested using long trajs; if 'load_part' == 'scale', SPGMT is tested on a large number of trajs.
                # check 'spatial_data_utils.py' for details.
                vali_node_list = dataload.load(load_part='test')
                # vali_node_ids, vali_node_list = dataload.load(load_part='long')
                s0 = time.time()
                embedding_vali = test_method.compute_spa_embedding(road_network=[road_network, distance_to_anchor_node], net=net,
                                                               test_traj=list(vali_node_list),
                                                               test_batch=self.test_batch)
                s1 = time.time()
                print("Embedding time: ", s1-s0)
                acc = test_method.test_spa_model(self.dataset, self.distance_type, embedding_vali, isvali=False)
                # acc = test_method.test_spa_model_long(self.dataset, self.distance_type, embedding_vali, vali_node_ids, isvali=False)
                s2 = time.time()
                print("Test time: ", s2-s1)
                print(acc)

    def Spa_train(self, load_model=None, load_optimizer=None):

        net = GraphTrajSimEncoder(feature_size=self.feature_size,
                                embedding_size=self.embedding_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                dropout_rate=self.dropout_rate,
                                concat=self.concat,
                                device=self.device,
                                usePE=self.usePE,
                                useSI=self.useSI,
                                dataset=self.dataset)

        dataload = spatial_data_utils.DataLoader()

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        lossfunction = SpaLossFun(self.train_batch, self.distance_type)

        net.to(self.device)
        lossfunction.to(self.device)

        if self.useSI:
            road_network = spatial_data_utils.load_neighbor(self.dataset)
            for item in road_network:
                item = item.to(self.device)
        else:
            road_network = spatial_data_utils.load_netowrk(self.dataset).to(self.device)
        distance_to_anchor_node = spatial_data_utils.sample_region_node(self.dataset).to(self.device)

        bt_num = int(dataload.return_triplets_num() / self.train_batch)

        batch_l = spatial_data_utils.batch_list(batch_size=self.train_batch)

        best_epoch = 0
        best_hr10 = 0
        lastepoch = '0'
        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            optimizer.load_state_dict(torch.load(load_optimizer))
            lastepoch = load_model.split('/')[-1].split('_')[3]
            best_epoch = int(lastepoch)

        accumulation_steps = int(128 / self.train_batch)
        for epoch in range(int(lastepoch), self.epochs):
            net.train()
            accumulation_loss = 0.0
            s1 = time.time()
            for bt in range(bt_num):
                a_node_batch, p_node_batch, n_node_batch, batch_index = batch_l.getbatch_one(shf=False)

                a_embedding = net([road_network, distance_to_anchor_node], a_node_batch)
                p_embedding = net([road_network, distance_to_anchor_node], p_node_batch)
                n_embedding = net([road_network, distance_to_anchor_node], n_node_batch)

                loss = lossfunction(a_embedding, p_embedding, n_embedding, batch_index)
                loss = loss / accumulation_steps
                accumulation_loss += loss

                loss.backward()

                if ((bt+1) % accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if (bt_num + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            s5 = time.time()
            # print("train time: ", s5 - s1)
            print('Epoch {}: Training time is {}, loss is {}'.format(epoch, (s5-s1), accumulation_loss))
            if epoch % 2 == 0:
                net.eval()
                with torch.no_grad():
                    s6 = time.time()
                    vali_node_list = dataload.load(load_part='vali')
                    embedding_vali = test_method.compute_spa_embedding(road_network=[road_network, distance_to_anchor_node], net=net,
                                                                   test_traj=list(vali_node_list),
                                                                   test_batch=self.test_batch)
                    acc = test_method.test_spa_model(self.dataset, self.distance_type, embedding_vali, isvali=True)
                    s7 = time.time()
                    print("test time: ", s7 - s6)
                    print(acc)

                    # save model
                    save_modelname = './model/{}_Spa/{}/{}_{}_epoch_{}_HR10_{}_HR50_{}_HR1050_{}_trainLoss_{}.pkl'.format(
                           self.dataset, self.distance_type, self.dataset, self.distance_type, str(epoch), acc[0], acc[1], acc[2], accumulation_loss.item())  # accumulation_loss.item()
                    torch.save(net.state_dict(), save_modelname)

                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                    # if epoch - best_epoch >= self.early_stop:
                    #     print(save_modelname)
                    #     break
