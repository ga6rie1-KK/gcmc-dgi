import torch
import torch.nn as nn
import torch.optim as optim
import argparse

def get_activation(act):
    if act == 'leaky':
        return nn.LeakyReLU(0.1)
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'softsign':
        return nn.Softsign()

def get_optimizer(opt):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam

def get_accumulation(accum):
    # input:tuple of tensors
    # output:accumulated tensors
    def sum(tensor_tuple):
        result = torch.zeros(tensor_tuple[0].shape)
        for tensor in tensor_tuple:
            result += tensor
        return result
    def stack(tensor_tuple):
        return torch.cat(tensor_tuple)
    if accum == 'sum':
        return sum
    elif accum == 'stack':
        return stack

def sparse_dropout(matrix, dropout_rate):
    idx = torch.nonzero(matrix, as_tuple=False)
    mask = torch.bernoulli(torch.ones(idx.shape[0]) * dropout_rate)
    for i in range(idx.shape[0]):
        if mask[i] == 0:
            matrix[idx[i][0], idx[i][1]] = 0
    return matrix

def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--msg_units', type=int, default=10)
    parser.add_argument('--out_units', type=int, default=5)
    parser.add_argument('--side_units', type=int, default=10)
    parser.add_argument('--use_sideFeat', type=bool, default=True)
    parser.add_argument('--dropout_rate', type=float, default=0.7)
    parser.add_argument('--Enc_act_func', default='relu')
    parser.add_argument('--Disc_act_func', default='relu')
    parser.add_argument('--sumvec_act_func', default='sigmoid')
    parser.add_argument('--accum_func', choices=['sum', 'stack'])
    parser.add_argument('--normalization', choices=['left', 'symmetric'])
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'])
    parser.add_argument('--max_iter', type=int, default=100)

    args = parser.parse_args()

    return args
