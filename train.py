import argparse
import os
import time
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable

from DSH import DSHLoss, dshnet
from dataset import getloader, getdshloader
from mAP import mAP

def create_network(net_name, codelength):
    if net_name == "dsh":
        net = dshnet(codelength=codelength)
    if net_name == "resnet18":
        net = models.resnet18(pretrained=True)
        net.fc = nn.Linear(512, codelength)
    return net.cuda()

def train():
    net.train()

    if netname == "dsh":
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005, nesterov=True)
    else:
        ignored_params = list(map(id, net.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             net.parameters())
        optimizer = torch.optim.SGD([
                    {'params': base_params},
                    {'params': net.fc.parameters(), 'lr': LR}
                ], lr=LR*0.1, momentum=0.9)

    for batch_idx, (data, target)in enumerate(trainloader):
        state['total_batch_num'] += 1
        data = Variable(data.cuda())
        target = Variable(target.cuda())

        # forward
        output = net(data)

        # backward
        optimizer.zero_grad()
        loss = lossmodule(output, target=target)
        loss.backward()
        optimizer.step()

        # calculate avg loss in a window
        state['loss_window'][state['total_batch_num'] % 20] = float(loss.data[0])
        loss_avg = sum(state['loss_window']) / float(20)
        state['loss_avg'] = loss_avg

        # log and print
        display = 50
        if (batch_idx + 1) % display == 0:
            toprint = '{}, LR: {}, epoch: {}, batch id: {}, avg_loss: {}, mAP: {}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
                LR, state['epoch'], batch_idx, round(loss_avg, 5), \
                round(state['map'], 4))
            print(toprint)

            with open(postfix+'.txt', 'a') as f:
                f.write(toprint + '\n')


def test():
    net.eval()
    outputs = []
    labels = []
    for batch_idx, (data, target)in enumerate(valloader):
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda(), volatile=True)

        output = net(data)

        outputs += output.data.cpu().tolist()
        labels += target.data.cpu().tolist()

    outputs = torch.Tensor(outputs)
    labels = torch.Tensor(labels)

    map_ = mAP(outputs, labels)
    print('map={}'.format(map_))
    state['map'] = map_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSH-Pytorch')
    parser.add_argument('--gpu_id', type=str, default='0', help="GPU ID")
    parser.add_argument('--dataset', type=str, default='cifar10', help="cifar10, cifar100 or imagenet100")
    parser.add_argument('--codelength', type=int, default=48, help="Hash code length")
    parser.add_argument('--net', type=str, default='dsh', help="network backbone type")
    parser.add_argument('--numclass_perbatch', type=int, default=0, help="num of classes for each mini-batch")
    parser.add_argument('--lr', type=float, default=0.01,  help="Initial learning rate")
    parser.add_argument('--postfix', type=str, default='', help="Indicator of this training")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 
    netname = args.net
    net = create_network(args.net, args.codelength)

    LR = args.lr
    tradeoff = 0.01
    codelength = args.codelength
    postfix = args.postfix
    dataset = args.dataset
    save_root = './{}_{}_len{}_{}'.format(args.net, args.dataset, args.codelength, args.postfix)
    if not os.path.exists(save_root):
        os.makedirs(os.path.join(save_root, 'snapshots'))

    with open(os.path.join(save_root, 'log.txt'), 'w') as f:
        f.write('\n')

    trainbatchsize = 200
    catenum = args.numclass_perbatch

    state = {}
    state['epoch'] = 0
    state['loss_avg'] = 0.0
    state['map'] = 0.0
    state['loss_window'] = [0] * 20
    state['total_batch_num'] = 0

    lossmodule = DSHLoss(bi_margin=2*codelength, tradeoff=tradeoff).cuda()
    if netname == "dsh":
        trainloader, valloader = getdshloader(trainbatchsize)
    else:
        trainloader, valloader = getloader(dataset, trainbatchsize, catenum)

    for epoch in range(150):
        state['epoch'] = epoch
        train()
        if (epoch+1) %30 == 0:
            test()
        torch.save(net.state_dict(), os.path.join(save_root, 'snapshots', 'epoch_{}.pytorch'.format(epoch),))
    test()
