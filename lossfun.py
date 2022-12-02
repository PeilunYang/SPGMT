from torch.nn import Module, Parameter
import torch
import numpy as np
import math
import yaml
# import pandas as pd
import torch.nn.functional as F


class STLossFun(Module):
    def __init__(self, train_batch, distance_type):
        super(STLossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        config = yaml.safe_load(open('config.yaml'))
        self.triplets_dis = np.load(str(config["path_triplets_truth"]))
        self.flag = True

        if self.flag:
            if str(config["dataset"]) == "tdrive" or "beijing":
                if str(config["distance_type"]) == "TP":
                    extra_coe = 0.5
                elif str(config["distance_type"]) == "DITA":
                    extra_coe = 1
                elif str(config["distance_type"]) == "LCRS":
                    extra_coe = 32
                elif str(config["distance_type"]) == "discret_frechet":
                    extra_coe = 1

            self.triplets_dis = self.triplets_dis * extra_coe

    def forward(self, embedding_a, embedding_p, embedding_n, batch_index):

        batch_triplet_dis = self.triplets_dis[batch_index]
        batch_loss = 0.0

        for i in range(self.train_batch):
            D_ap = math.exp(-batch_triplet_dis[i][0])
            D_an = math.exp(-batch_triplet_dis[i][1])

            v_ap = torch.exp(-(torch.dist(embedding_a[i], embedding_p[i], p=2)))
            v_an = torch.exp(-(torch.dist(embedding_a[i], embedding_n[i], p=2)))

            loss_entire_ap = (D_ap - v_ap) ** 2
            loss_entire_an = (D_an - v_an) ** 2

            if D_ap > D_an:
                oneloss = loss_entire_ap + loss_entire_an + (F.relu(v_an - v_ap)) ** 2
            else:
                oneloss = loss_entire_ap + loss_entire_an
            batch_loss += oneloss

        mean_batch_loss = batch_loss / self.train_batch
        sum_batch_loss = batch_loss

        return mean_batch_loss


class SpaLossFun(Module):
    def __init__(self,train_batch,distance_type):
        super(SpaLossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        config = yaml.safe_load(open('config.yaml'))
        self.triplets_dis = np.load(str(config["spatial_path_triplets_truth"]))
        self.flag = True

        if self.flag:
            if str(config["dataset"]) == "tdrive" or "beijing" or "porto":
                if str(config["distance_type"]) == "TP":
                    extra_coe = 4
                elif str(config["distance_type"]) == "DITA":
                    if str(config["dataset"]) == "beijing":
                        extra_coe = 8
                    if str(config["dataset"]) == "porto":
                        extra_coe = 4
                elif str(config["distance_type"]) == "LCRS":
                    extra_coe = 16
                elif str(config["distance_type"]) == "discret_frechet":
                    extra_coe = 4

            self.triplets_dis = self.triplets_dis * extra_coe

    def forward(self, embedding_a, embedding_p, embedding_n, batch_index):

        batch_triplet_dis = self.triplets_dis[batch_index]
        batch_loss = 0.0

        for i in range(self.train_batch):
            D_ap = math.exp(-batch_triplet_dis[i][0])
            D_an = math.exp(-batch_triplet_dis[i][1])

            v_ap = torch.exp(-(torch.dist(embedding_a[i], embedding_p[i], p=2)))
            v_an = torch.exp(-(torch.dist(embedding_a[i], embedding_n[i], p=2)))

            loss_entire_ap = (D_ap - v_ap) ** 2
            loss_entire_an = (D_an - v_an) ** 2

            if D_ap > D_an:
                oneloss = loss_entire_ap + loss_entire_an + (F.relu(v_an - v_ap)) ** 2
            else:
                oneloss = loss_entire_ap + loss_entire_an
            batch_loss += oneloss

        mean_batch_loss = batch_loss / self.train_batch
        sum_batch_loss = batch_loss

        return mean_batch_loss

