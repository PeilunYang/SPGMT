import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import yaml

def compute_embedding(net, road_network, test_traj, test_time, test_batch):

    if len(test_traj) <= test_batch:
        embedding = net(road_network, test_traj, test_time)
        return embedding
    else:
        i = 0
        all_embedding = []
        while i < len(test_traj):
            embedding = net(road_network, test_traj[i:i+test_batch], test_time[i:i+test_batch])
            all_embedding.append(embedding)
            i += test_batch

        all_embedding = torch.cat(all_embedding,0)
        return all_embedding


def compute_spa_embedding(net, road_network, test_traj, test_batch):

    if len(test_traj) <= test_batch:
        embedding = net(road_network, test_traj)
        return embedding
    else:
        i = 0
        all_embedding = []
        while i < len(test_traj):
            embedding = net(road_network, test_traj[i:i+test_batch])
            all_embedding.append(embedding)
            i += test_batch

        all_embedding = torch.cat(all_embedding,0)
        return all_embedding


def test_model(dataset, distance_type, embedding_set, isvali=False):
    config = yaml.safe_load(open('config.yaml'))
    if isvali==True:
        # input_dis_matrix = np.load(str(config["path_vali_truth"]))
        input_dis_matrix = np.load('./ground_truth/' + dataset + '/' + distance_type + '/vali_st_distance.npy')
        total_test_num = config["vali_set_size"] - config["train_set_size"]
    else:
        # input_dis_matrix = np.load(str(config["path_test_truth"]))
        input_dis_matrix = np.load('./ground_truth/' + dataset + '/' + distance_type + '/test_st_distance.npy')[:, :15800]
        total_test_num = config["test_set_size"] - config["vali_set_size"]

    embedding_set = embedding_set.data.cpu().numpy()[:15800, :]
    print(embedding_set.shape)

    embedding_dis_matrix = []
    for t in embedding_set:
        emb = np.repeat([t], repeats=len(embedding_set), axis=0)
        matrix = np.linalg.norm(emb-embedding_set, ord=2, axis=1)
        embedding_dis_matrix.append(matrix.tolist())

    l_recall_10 = 0
    l_recall_50 = 0
    l_recall_10_50 = 0
    l_top1_hr1 = 0
    l_top1_hr10 = 0
    l_top1_hr50 = 0

    f_num = 0

    for i in range(len(input_dis_matrix)):
        input_r = np.array(input_dis_matrix[i])
        one_index = []
        for idx, value in enumerate(input_r):
            if value != -1:
                one_index.append(idx)
        input_r = input_r[one_index]
        input_r = input_r[:total_test_num]

        input_r50 = np.argsort(input_r)[1:51]
        input_r10 = input_r50[:10]

        embed_r = np.array(embedding_dis_matrix[i])
        embed_r = embed_r[one_index]
        embed_r = embed_r[:total_test_num]

        embed_r50 = np.argsort(embed_r)[1:51]
        embed_r10 = embed_r50[:10]

        if len(one_index)>=51:
            f_num += 1
            l_top1_hr1 += len(list(set(input_r10[:1]).intersection(set(embed_r10[:1]))))
            l_top1_hr10 += len(list(set(input_r10[:1]).intersection(set(embed_r10))))
            l_top1_hr50 += len(list(set(input_r10[:1]).intersection(set(embed_r50))))
            l_recall_10 += len(list(set(input_r10).intersection(set(embed_r10))))
            l_recall_50 += len(list(set(input_r50).intersection(set(embed_r50))))
            l_recall_10_50 += len(list(set(input_r50).intersection(set(embed_r10))))

    recall_10 = float(l_recall_10) / (10 * f_num)
    recall_50 = float(l_recall_50) / (50 * f_num)
    recall_10_50 = float(l_recall_10_50) / (10 * f_num)
    top1_hr1 = float(l_top1_hr1) / (1 * f_num)
    top1_hr10 = float(l_top1_hr10) / (1 * f_num)
    top1_hr50 = float(l_top1_hr50) / (1 * f_num)
    print('Dataset: {}, Distance type: {}, f_num is {}'.format(dataset, distance_type, f_num))

    return recall_10, recall_50, recall_10_50, top1_hr1, top1_hr10, top1_hr50


def test_spa_model(dataset, distance_type, embedding_set, isvali=False):
    config = yaml.safe_load(open('config.yaml'))
    if isvali==True:
        # input_dis_matrix = np.load(str(config["spatial_path_vali_truth"]))
        input_dis_matrix = np.load('./ground_truth/' + dataset + '/' + distance_type + '/vali_spa_distance.npy')
        print('./ground_truth/' + dataset + '/' + distance_type + '/vali_spa_distance.npy')
        total_test_num = config["vali_set_size"] - config["train_set_size"]
    else:
        # input_dis_matrix = np.load(str(config["spatial_path_test_truth"]))
        input_dis_matrix = np.load('./ground_truth/' + dataset + '/' + distance_type + '/test_spa_distance.npy')
        print('./ground_truth/' + dataset + '/' + distance_type + '/test_spa_distance.npy')
        total_test_num = config["test_set_size"] - config["vali_set_size"]

    embedding_set = embedding_set.data.cpu().numpy()
    print(embedding_set.shape)

    embedding_dis_matrix = []
    for t in embedding_set:
        emb = np.repeat([t], repeats=len(embedding_set), axis=0)
        matrix = np.linalg.norm(emb-embedding_set, ord=2, axis=1)
        embedding_dis_matrix.append(matrix.tolist())

    l_recall_10 = 0
    l_recall_50 = 0
    l_recall_10_50 = 0
    l_mse_loss = 0
    l_top1_hr1 = 0
    l_top1_hr10 = 0
    l_top1_hr50 = 0

    f_num = 0

    for i in range(len(input_dis_matrix)):
        input_r = np.array(input_dis_matrix[i])
        one_index = []
        max_index = []
        for idx, value in enumerate(input_r):
            if value != -1:
                one_index.append(idx)
            if value == 1.0:
                max_index.append(idx)
        input_r = input_r[one_index]
        input_r = input_r[:total_test_num]  # input_r = input_r[:12000]

        input_r50 = np.argsort(input_r)[1:51]
        input_r10 = input_r50[:10]

        embed_r = np.array(embedding_dis_matrix[i])
        embed_r = embed_r[one_index]
        embed_r = embed_r[:total_test_num]  # embed_r = embed_r[:12000]

        embed_r50 = np.argsort(embed_r)[1:51]
        embed_r10 = embed_r50[:10]

        if len(one_index) >= 51:
            f_num += 1
            l_top1_hr1 += len(list(set(input_r10[:1]).intersection(set(embed_r10[:1]))))
            l_top1_hr10 += len(list(set(input_r10[:1]).intersection(set(embed_r10))))
            l_top1_hr50 += len(list(set(input_r10[:1]).intersection(set(embed_r50))))
            if len(max_index) > (total_test_num - 11):
                tmp_cnt = total_test_num - len(max_index)
                l_recall_10 += len(list(set(input_r10[:tmp_cnt]).intersection(set(embed_r10[:tmp_cnt]))))
                l_recall_50 += (len(list(set(input_r10[:tmp_cnt]).intersection(set(embed_r10[:tmp_cnt])))) + 50 - tmp_cnt)
                # l_recall_10_50 += (len(list(set(input_r10[:tmp_cnt]).intersection(set(embed_r10[:tmp_cnt])))) + 10 - tmp_cnt)
                l_recall_10_50 += 10
                # l_mse_loss += np.linalg.norm(input_r[1:tmp_cnt] - embed_r[1:tmp_cnt])
            elif (total_test_num - 11) > len(max_index) and len(max_index) > (total_test_num - 51):
                tmp_cnt = total_test_num - len(max_index)
                l_recall_10 += len(list(set(input_r10).intersection(set(embed_r10))))
                l_recall_50 += (len(list(set(input_r50[:tmp_cnt]).intersection(set(embed_r50[:tmp_cnt])))) + 50 - tmp_cnt)
                if tmp_cnt >= 40:
                    l_recall_10_50 += (len(list(set(input_r50[:tmp_cnt]).intersection(set(embed_r10)))) + 50 - tmp_cnt)
                else:
                    l_recall_10_50 += 10
                # l_mse_loss += np.linalg.norm(input_r[1:tmp_cnt] - embed_r[1:tmp_cnt])
            else:
                l_recall_10 += len(list(set(input_r10).intersection(set(embed_r10))))
                l_recall_50 += len(list(set(input_r50).intersection(set(embed_r50))))
                l_recall_10_50 += len(list(set(input_r50).intersection(set(embed_r10))))
                # l_mse_loss += np.linalg.norm(input_r[1:]-embed_r[1:])

    recall_10 = float(l_recall_10) / (10 * f_num)
    recall_50 = float(l_recall_50) / (50 * f_num)
    recall_10_50 = float(l_recall_10_50) / (10 * f_num)
    # mse_loss = float(l_mse_loss / f_num)
    top1_hr1 = float(l_top1_hr1) / (1 * f_num)
    top1_hr10 = float(l_top1_hr10) / (1 * f_num)
    top1_hr50 = float(l_top1_hr50) / (1 * f_num)
    print('Dataset: {}, Distance type: {}, f_num is {}'.format(dataset, distance_type, f_num))

    return recall_10, recall_50, recall_10_50, top1_hr1, top1_hr10, top1_hr50  # , mse_loss


def test_spa_model_long(dataset, distance_type, embedding_set, long_ids, isvali=False):
    config = yaml.safe_load(open('config.yaml'))
    if isvali==True:
        # input_dis_matrix = np.load(str(config["spatial_path_vali_truth"]))
        input_dis_matrix = np.load('./ground_truth/' + dataset + '/' + distance_type + '/vali_spa_distance.npy')
        total_test_num = config["vali_set_size"] - config["train_set_size"]
    else:
        # input_dis_matrix = np.load(str(config["spatial_path_test_truth"]))
        input_dis_matrix = np.load('./ground_truth/' + dataset + '/' + distance_type + '/test_spa_distance.npy')
        total_test_num = config["test_set_size"] - config["vali_set_size"]

    embedding_set = embedding_set.data.cpu().numpy()
    # print(embedding_set.shape)

    embedding_set = embedding_set[long_ids, :]
    input_dis_matrix = input_dis_matrix[long_ids, :][:, long_ids]
    print(input_dis_matrix.shape)
    print(embedding_set.shape)

    embedding_dis_matrix = []
    for t in embedding_set:
        emb = np.repeat([t], repeats=len(embedding_set), axis=0)
        matrix = np.linalg.norm(emb-embedding_set, ord=2, axis=1)
        embedding_dis_matrix.append(matrix.tolist())

    l_recall_10 = 0
    l_recall_50 = 0
    l_recall_10_50 = 0
    l_mse_loss = 0
    l_top1_hr1 = 0
    l_top1_hr10 = 0
    l_top1_hr50 = 0

    f_num = 0

    for i in range(len(input_dis_matrix)):
        input_r = np.array(input_dis_matrix[i])
        one_index = []
        max_index = []
        for idx, value in enumerate(input_r):
            if value != -1:
                one_index.append(idx)
            if value == 1.0:
                max_index.append(idx)
        input_r = input_r[one_index]
        input_r = input_r[:total_test_num]  # input_r = input_r[:12000]

        input_r50 = np.argsort(input_r)[1:51]
        input_r10 = input_r50[:10]

        embed_r = np.array(embedding_dis_matrix[i])
        embed_r = embed_r[one_index]
        embed_r = embed_r[:total_test_num]  # embed_r = embed_r[:12000]

        embed_r50 = np.argsort(embed_r)[1:51]
        embed_r10 = embed_r50[:10]

        if len(one_index) >= 51:
            f_num += 1
            l_top1_hr1 += len(list(set(input_r10[:1]).intersection(set(embed_r10[:1]))))
            l_top1_hr10 += len(list(set(input_r10[:1]).intersection(set(embed_r10))))
            l_top1_hr50 += len(list(set(input_r10[:1]).intersection(set(embed_r50))))
            if len(max_index) > (total_test_num - 11):
                tmp_cnt = total_test_num - len(max_index)
                l_recall_10 += len(list(set(input_r10[:tmp_cnt]).intersection(set(embed_r10[:tmp_cnt]))))
                l_recall_50 += (len(list(set(input_r10[:tmp_cnt]).intersection(set(embed_r10[:tmp_cnt])))) + 50 - tmp_cnt)
                # l_recall_10_50 += (len(list(set(input_r10[:tmp_cnt]).intersection(set(embed_r10[:tmp_cnt])))) + 10 - tmp_cnt)
                l_recall_10_50 += 10
                # l_mse_loss += np.linalg.norm(input_r[1:tmp_cnt] - embed_r[1:tmp_cnt])
            elif (total_test_num - 11) > len(max_index) and len(max_index) > (total_test_num - 51):
                tmp_cnt = total_test_num - len(max_index)
                l_recall_10 += len(list(set(input_r10).intersection(set(embed_r10))))
                l_recall_50 += (len(list(set(input_r50[:tmp_cnt]).intersection(set(embed_r50[:tmp_cnt])))) + 50 - tmp_cnt)
                if tmp_cnt >= 40:
                    l_recall_10_50 += (len(list(set(input_r50[:tmp_cnt]).intersection(set(embed_r10)))) + 50 - tmp_cnt)
                else:
                    l_recall_10_50 += 10
                # l_mse_loss += np.linalg.norm(input_r[1:tmp_cnt] - embed_r[1:tmp_cnt])
            else:
                l_recall_10 += len(list(set(input_r10).intersection(set(embed_r10))))
                l_recall_50 += len(list(set(input_r50).intersection(set(embed_r50))))
                l_recall_10_50 += len(list(set(input_r50).intersection(set(embed_r10))))
                # l_mse_loss += np.linalg.norm(input_r[1:]-embed_r[1:])

    recall_10 = float(l_recall_10) / (10 * f_num)
    recall_50 = float(l_recall_50) / (50 * f_num)
    recall_10_50 = float(l_recall_10_50) / (10 * f_num)
    # mse_loss = float(l_mse_loss / f_num)
    top1_hr1 = float(l_top1_hr1) / (1 * f_num)
    top1_hr10 = float(l_top1_hr10) / (1 * f_num)
    top1_hr50 = float(l_top1_hr50) / (1 * f_num)
    print('Dataset: {}, Distance type: {}, f_num is {}'.format(dataset, distance_type, f_num))

    return recall_10, recall_50, recall_10_50, top1_hr1, top1_hr10, top1_hr50  # , mse_loss