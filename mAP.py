import torch
import time
from torch.autograd import Variable
import numpy as np

# bicode = torch.FloatTensor([[1,-1,1],[-1,-1,1],[-1,-1,-1]])
# label = torch.FloatTensor([1,3,1])

# bicode = torch.rand(10000,24)-0.5
# label = np.random.randint(100,size=10000)
# label = torch.from_numpy(label)

# import scipy.io as scio
# bicode = torch.from_numpy(scio.loadmat('code.mat')['B']).type(torch.LongTensor)
# label = torch.from_numpy(scio.loadmat('label.mat')['label']).view(-1).type(torch.LongTensor)

# bicode = Variable(bicode).cuda()
# label = Variable(label).cuda()

def mAP(bicode, label):
    if type(bicode)==torch.autograd.variable.Variable:
        bicode=bicode.data
    if type(label)==torch.autograd.variable.Variable:
        label=label.data
    bicode = torch.sign(bicode.cpu()).type(torch.FloatTensor)
    label = label.cpu()
    sampleNum = label.size(0)

    hashingDistance = -torch.mm(bicode, bicode.transpose(0,1))
    sorted, indices = torch.sort(hashingDistance)
    sorted_distance_labels = torch.gather(torch.stack([label]*sampleNum),1,indices)

    sorted_sim_matrix = (torch.stack([label]*sampleNum).transpose(0,1)==sorted_distance_labels).type(torch.FloatTensor)
    sorted_sim_matrix[:,0] = 0
    rank_matrix = torch.mm(sorted_sim_matrix.type(torch.FloatTensor), torch.FloatTensor(np.triu(np.ones([sampleNum,sampleNum])))) * sorted_sim_matrix

    tmp = torch.FloatTensor([list(range(sampleNum))]*sampleNum)
    tmp[:,0] = 1
    mapvalue = ((rank_matrix/tmp).sum(1)/(sorted_sim_matrix.sum(1)+1e-7)).sum()/float(sampleNum)
    return mapvalue

# cuda verison is smaller than cpu version 
# have no idea why this happend now
def mAP_cuda(bicode, label):
    bicode = torch.sign(bicode).type(torch.cuda.FloatTensor)
    laebl = label.type(torch.cuda.FloatTensor)


    sampleNum = label.size(0)

    hashingDistance = -torch.mm(bicode, bicode.transpose(0,1))
    sorted, indices = torch.sort(hashingDistance)

    sorted_distance_labels = torch.gather(torch.stack([label]*sampleNum),1,indices)

    sorted_sim_matrix = (torch.stack([label]*sampleNum).transpose(0,1)==sorted_distance_labels).type(torch.cuda.FloatTensor)
    sorted_sim_matrix[:,0] = 0

    rank_matrix = torch.mm(sorted_sim_matrix, Variable(torch.FloatTensor(np.triu(np.ones([sampleNum,sampleNum])))).cuda()) * sorted_sim_matrix

    tmp = Variable(torch.FloatTensor([list(range(sampleNum))]*sampleNum)).cuda()
    tmp[:,0] = 1
    mapvalue = ((rank_matrix/tmp).sum(1)/(sorted_sim_matrix.sum(1)+1e-7)).sum()/float(sampleNum)
    mapvalue = mapvalue.cpu().data[0]

    return mapvalue
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
def euclidean_mAP(bicode, label):
    if type(bicode)==torch.autograd.variable.Variable:
        bicode=bicode.data
    if type(label)==torch.autograd.variable.Variable:
        label=label.data
    bicode = bicode.type(torch.FloatTensor)
    label = label.cpu()
    sampleNum = label.size(0)

    euclideanDistance = pairwise_distances(bicode)
    sorted, indices = torch.sort(euclideanDistance)
    sorted_distance_labels = torch.gather(torch.stack([label]*sampleNum),1,indices)

    sorted_sim_matrix = (torch.stack([label]*sampleNum).transpose(0,1)==sorted_distance_labels).type(torch.FloatTensor)
    sorted_sim_matrix[:,0] = 0
    rank_matrix = torch.mm(sorted_sim_matrix.type(torch.FloatTensor), torch.FloatTensor(np.triu(np.ones([sampleNum,sampleNum])))) * sorted_sim_matrix

    tmp = torch.FloatTensor([list(range(sampleNum))]*sampleNum)
    tmp[:,0] = 1
    mapvalue = ((rank_matrix/tmp).sum(1)/(sorted_sim_matrix.sum(1)+1e-7)).sum()/float(sampleNum)
    return mapvalue
# mapvalue = mAP(bicode, label)