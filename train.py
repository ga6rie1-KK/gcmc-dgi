import torch
import torch.nn as nn
from model import Encoder, Decoder, Discriminator
from utils import get_activation, get_optimizer, config
from data import DataSet

class Net(nn.Module):
    def __init__(self, args, dataset):
        super(Net, self).__init__()
        self.encoder = Encoder(dataset.adj_matrix,
                               dataset.num_rating,
                               dataset.u_sidefeat,
                               dataset.v_sidefeat,
                               args.msg_units,
                               args.out_units,
                               args.side_units,
                               args.Enc_act_func,
                               args.accum_func,
                               args.normalization,
                               args.use_sideFeat,
                               args.dropout_rate)
        self.decoder = Decoder(dataset.num_rating,
                               args.out_units,
                               args.dropout_rate)
        self.discriminator = Discriminator(args.out_units,
                                           args.Disc_act_func)
        self.sumvec_act = get_activation(args.sumvec_act_func)

    def forward(self):
        # encode positive graph
        user_emb, item_emb = self.encoder(False)
        posi_feat = torch.cat((user_emb, item_emb), 0)
        # encode negative graph
        nega_user_emb, nega_item_emb = self.encoder(True)
        nega_feat = torch.cat((nega_user_emb, nega_item_emb), 0)
        # prepare summary vector
        sum_vec = torch.sum(torch.t(posi_feat), 1) / posi_feat.shape[0]
        sum_vec = self.sumvec_act(sum_vec)
        # reconstruct rating matrix
        pred_rate_prob = self.decoder(user_emb, item_emb)
        # diecriminate positive and negative graph
        posi_disc = self.discriminator(posi_feat, sum_vec)
        nega_disc = self.discriminator(nega_feat, sum_vec)

        return pred_rate_prob, posi_disc, nega_disc


def train():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DataSet()
    # dataset.to(device)
    net = Net(args, dataset)
    # net.to(device)
    adjs = net.encoder.adjs

    def criterion(adj_matrixs, pred_rate_prob, posi_disc, nega_disc):
        reconstruct_error = 0
        for r in range(len(adj_matrixs)):
            reconstruct_error += torch.sum(adj_matrixs[r] * torch.log(pred_rate_prob[r]))
        discrimination_error = torch.mean(torch.log(posi_disc) + torch.log(1 - posi_disc))/2
        return - reconstruct_error - discrimination_error

    learning_rate = args.learning_rate
    optimizer = get_optimizer(args.optimizer)(net.parameters(), lr=learning_rate)
    for epoch in range(args.max_iter):
        net.train()
        pred_rate_prob, posi_disc, nega_disc = net()
        optimizer.zero_grad()
        loss = criterion(adjs, pred_rate_prob, posi_disc, nega_disc)
        loss.backward()
        optimizer.step()
        print(loss)

        # # predict ratings
        # rating_matrix = torch.zeros(num_user, num_item)
        # for r in range(dataset['num_rating']):
        #     rating_matrix += r * pred_rate_prob[r]
        # rmse = ((dataset['adj_matrix'] - rating_matrix) ** 2).sum()

args = config()
train()
