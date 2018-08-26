# The LRN layer in the previous model is replaced by nn.Batchnorm2d module in this implementation

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import init

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class DSH(nn.Module):

    def __init__(self, codelength=12):
        super(DSH, self).__init__()
        self.codelength = codelength
        
        # the input shape if linear1 is calculated manually
        # shape after conv layer or pooling layer  = (input_width+2*pad-pool_size)/stride+1
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('batchnorm1', nn.BatchNorm2d(32)),
            ('conv2', nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=True)),
            ('pool2', nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ('relu2', nn.ReLU(inplace=True)),
            ('batchnorm2', nn.BatchNorm2d(32)),
            ('conv3', nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True))
        ]))
        self.linear = nn.Sequential(OrderedDict([
            # ('drop', nn.Dropout()),
            ('linear1', nn.Linear(64 * 4 * 4, 500)),
            ('relu', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(500, self.codelength))
        ]))
 


    def forward(self, x):
        out = self.features(x)
        out = self.linear(out.view(x.size(0), -1))
        return out


def getMask(length):
    # return an upper triangular matrix
    mask = np.ones([length, length])
    mask = 1 - torch.Tensor(np.triu(mask))
    mask = Variable(mask.cuda())
    return mask


def distanceMatrix(mtx1, mtx2):
    m = mtx1.size(0)
    p = mtx1.size(1)
    mmtx1 = torch.stack([mtx1] * m)
    mmtx2 = torch.stack([mtx2] * m).transpose(0, 1)
    dist = torch.sum((mmtx1 - mmtx2) ** 2, 2).squeeze()
    return dist


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def labelToSim(label):
    length = len(label)
    l1 = torch.stack([label] * length)
    simMatrix = (l1 == l1.transpose(0, 1)).type(torch.cuda.FloatTensor)
    return simMatrix


class DSHLoss(nn.Module):

    def __init__(self, bi_margin=24, tradeoff=0.01):
        super(DSHLoss, self).__init__()
        self.bi_margin = bi_margin
        self.tradeoff = tradeoff
        # import os
        # self.log='loss_tmp.txt'
        # if os.path.exists(self.log):
        #     raise Exception("loss_tmp file already exists!")
        # with open(self.log, 'w') as f:
        #     f.write('simloss'.rjust(10) + '\t' + 'notsimloss'.rjust(10) + '\t' + 'quanloss'.rjust(10) + '\n')

    def forward(self, input, target):
        batchsize = input.size(0)
        codelength = input.size(1)

        D = pairwise_distances(input, input)
        S = labelToSim(target)

        inputabs = input.abs()
        ones = Variable(torch.ones(batchsize, codelength).cuda())
        input_ones_dist = (inputabs - ones).abs().sum(1)
        input_ones_dist = torch.stack([input_ones_dist] * batchsize)
        input_ones_dist = input_ones_dist + input_ones_dist.transpose(0, 1)
        input_ones_dist = input_ones_dist.abs()

        mask = getMask(batchsize)

        threshold_module = nn.Threshold(0, 0)
        notsimloss = threshold_module(self.bi_margin - D)
        loss = 0.5 * D * S + 0.5 * notsimloss * (1 - S) + self.tradeoff * (input_ones_dist)
        loss1 = ((0.5 * D * S * mask).sum() / mask.sum()).data[0]
        loss2 = ((0.5 * notsimloss * (1 - S) * mask).sum() / mask.sum()).data[0]
        loss3 = ((self.tradeoff * (input_ones_dist) * mask).sum() / mask.sum()).data[0]
        # with open(self.log, 'a') as f:
        #     f.write((('%.4f ' % (loss1)).rjust(15)) + '\t' + ('%.4f ' % (loss2)).rjust(15) + '\t' + (
        #             '%.4f ' % (loss3)).rjust(15) + '\n')
        loss = loss * mask
        loss = loss.sum() / mask.sum()
        return loss


def dshnet(state_dict=None, **kwargs):
    net = DSH(**kwargs)
    own_state = net.state_dict()
    if state_dict is not None:
        for name, param in state_dict.items():
            if name not in own_state:
                continue

            try:
                own_state[name].copy_(param)
            except:
                print('{} layer parameters not compatitable'.format(name))
                if 'weight' in name:
                    # init.kaiming_normal(own_state[name])
                    init.xavier_normal(own_state[name])
                else:
                    own_state[name].zero_()
    else:
        for name, param in own_state.items():
            if 'weight' in name:
                if 'batchnorm' in name:
                    own_state[name].fill_(1)
                else:
                    init.xavier_normal(own_state[name])
            else:
                own_state[name].zero_()

    return net

# from torch.autograd import Variable
# net = DSH()
# x = Variable(torch.rand(8,3,32,32))
# feas = net.features(x)

# label = Variable(torch.Tensor([1,3,3,5,6,5,1,6]))
# dshm = dshnet()
# out = dshm(x)
# a = pairwise_distances(out)
# loss_module = DSHLoss()
# loss = loss_module(out, label)


