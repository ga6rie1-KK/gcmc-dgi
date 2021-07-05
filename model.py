import torch
import torch.nn as nn
import numpy as np

from utils import get_activation, get_accumulation, sparse_dropout

# assuming ratings are int
class Encoder(nn.Module):
    def __init__(self,
                 adj_matrix, num_rating, u_sideFeat, v_sideFeat,
                 msg_units, out_units, side_units,
                 act_func, accum_func, normalization, use_sideFeat,
                 dropout_rate):
        # adj_matrix:user×item, u,v_sideFeat:user,item×dim

        super(Encoder, self).__init__()

        self.num_user = adj_matrix.shape[0]
        self.num_item = adj_matrix.shape[1]
        self.num_rating = num_rating

        self.use_sideFeat = use_sideFeat
        self.msg_units = msg_units
        self.out_units = out_units
        if self.use_sideFeat == True:
            self.side_units = side_units

        # devide rating matrix with rating
        self.adjs = [torch.where(adj_matrix == r+1,
                                 torch.ones(self.num_user, self.num_item),
                                 torch.zeros(self.num_user, self.num_item))
                     for r in range(num_rating)]

        # prepare one hot vector features
        # user,item×dim
        I = torch.eye(self.num_user + self.num_item)
        self.user_1hot = I[:self.num_user, :]
        self.item_1hot = I[self.num_user:, :]

        if self.use_sideFeat == True:
            # prepare side features and side layer
            self.u_sideFeat = u_sideFeat
            self.v_sideFeat = v_sideFeat
            self.u_sideLayer1 = nn.Linear(u_sideFeat.shape[1], side_units, True)
            self.u_sideLayer2 = nn.Linear(side_units, out_units, False)
            self.v_sideLayer1 = nn.Linear(v_sideFeat.shape[1], side_units, True)
            self.v_sideLayer2 = nn.Linear(side_units, out_units, False)

        # prepare message layer
        # stuck割り切れないとき要検討
        if accum_func == "sum":
            self.msgLayers = nn.ModuleList([nn.Linear(self.num_user+self.num_item,  msg_units, False)
                                            for r in range(num_rating)])
        if accum_func == "stack":
            self.msgLayers = nn.ModuleList([nn.Linear(self.num_user+self.num_item,  int(msg_units/num_rating), False)
                                            for r in range(num_rating)])


        # prepare normalizer
        # Cu:user×item, Cv:item×user
        Nu = torch.squeeze(sum([torch.sum(adjr, 1, True) for adjr in self.adjs]))
        Nv = torch.squeeze(sum([torch.sum(torch.t(adjr), 1, True) for adjr in self.adjs]))
        if normalization == "left":
            self.Cu = torch.t(torch.stack([Nu for i in range(self.num_item)]))
            self.Cv = torch.t(torch.stack([Nv for i in range(self.num_user)]))
        if normalization == "symmetric":
            C = torch.sqrt(torch.t(torch.unsqueeze(Nu, 0)) @ torch.unsqueeze(Nv, 0))
            self.Cu = C
            self.Cv = torch.t(C)

        # prepare accumulation function
        self.accum = get_accumulation(accum_func)

        # prepare dense layer
        self.denseLayer = nn.Linear(msg_units, out_units, False)

        # prepare activtion function
        self.act = get_activation(act_func)

        # prepare doupout function
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, isNegative):
        user_1hot = self.user_1hot
        item_1hot = self.item_1hot
        if isNegative:
            user_idx = np.random.permutation(self.num_user)
            item_idx = np.random.permutation(self.num_item)
            user_1hot = user_1hot[user_idx, :]
            item_1hot = item_1hot[item_idx, :]
        # sparse dropout
        user_1hot = sparse_dropout(user_1hot, self.dropout_rate)
        item_1hot = sparse_dropout(item_1hot, self.dropout_rate)

        # message layer
        Hu = torch.empty(self.num_user, self.msg_units)
        for i in range(self.num_user):
            lst = []
            for r in range(self.num_rating):
                lst.append(self.msgLayers[r](torch.sum(torch.t(torch.diag(torch.mul(self.adjs[r][i,:], 1/self.Cu[i])) @ item_1hot), 1)))
            Hu[i,:] = self.act(self.accum(tuple(lst)))
        Hv = torch.empty(self.num_item, self.msg_units)
        for j in range(self.num_item):
            lst = []
            for r in range(self.num_rating):
                lst.append(self.msgLayers[r](torch.sum(torch.t(torch.diag(torch.mul(self.adjs[r][:,j], 1/self.Cv[j])) @ user_1hot), 1)))
            Hv[j,:] = self.act(self.accum(tuple(lst)))

        # dropout
        Hu = self.dropout(Hu)
        Hv = self.dropout(Hv)

        if self.use_sideFeat == False:
            # dense layer
            U = self.act(self.denseLayer(Hu))
            V = self.act(self.denseLayer(Hv))

        if self.use_sideFeat == True:
            # side layer
            Fu = self.act(self.u_sideLayer1(self.u_sideFeat))
            Fv = self.act(self.v_sideLayer1(self.v_sideFeat))
            # dense layer
            U = self.act(self.denseLayer(Hu) + self.u_sideLayer2(Fu))
            V = self.act(self.denseLayer(Hv) + self.v_sideLayer2(Fv))

        return U, V


class Decoder(nn.Module):
    def __init__(self, num_rating, input_units, dropout_rate):
        super(Decoder, self).__init__()

        self.num_rating = num_rating

        # prepare Q
        self.params = nn.ParameterList()
        for r in range(self.num_rating):
            self.params.append(nn.Parameter(torch.randn(input_units, input_units)))

        # prepare doupout function
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ufeat, vfeat):
        # dropout
        ufeat = self.dropout(ufeat)
        vfeat = self.dropout(vfeat)

        # calculate probability
        similarity = []
        similarity_sum = torch.zeros(ufeat.shape[0], vfeat.shape[0])
        for r in range(self.num_rating):
            sim = torch.exp(ufeat @ self.params[r] @ torch.t(vfeat))
            similarity.append(sim)
            similarity_sum += sim
        probability = [similarity[r]/similarity_sum for r in range(self.num_rating)]

        # # calculate expectation
        # rating_matrix = torch.zeros(ufeat.shape[0], vfeat.shape[0])
        # for i in range(ufeat.shape[0]):
        #     for j in range(vfeat.shape[0]):
        #         for r in range(self.num_rating):
        #             rating_matrix[i][j] += r * similarity[r][i][j]
        #         rating_matrix[i][j] = rating_matrix[i][j] / similarity_sum[i][j]
        #
        # return rating_matrix, probability

        return probability


# output：[D(h_1,s),...,D(h_n,s)]^T
class Discriminator(nn.Module):
    def __init__(self, input_units, act_func):
        super(Discriminator, self).__init__()

        # prepare W
        self.parameter = nn.Parameter(torch.randn(input_units, input_units))
        # prepare activation function
        self.act = get_activation(act_func)

    def forward(self, feat, sum_vec):
        # feat：N×dim, sum_vec：1×dim
        similarity = self.act(feat @ self.parameter @ torch.t(sum_vec))

        return similarity
