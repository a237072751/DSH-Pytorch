import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import json
import sampler
import sampler2
import cifarDataset

default_transform = transforms.Compose([transforms.ToTensor()])


def default_loader(path, transform):
    img = Image.open(path).convert('RGB')
    return transform(img)



class imagenet(data.Dataset):
    def __init__(self, root, train,transform=default_transform):
        """
        :param root: image path root
        :param train:  path contain the imagenames and its labels e.g.: './train.txt'
        :param transform:
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.part_paths = []
        self.labels = []
        with open(self.train) as f:
            for line in f:
                items = line.split()
                part_path = items[0].strip()
                wnid = items[1].strip()
                self.part_paths.append(part_path)
                self.labels.append(wnid)
        self.length = len(self.labels)

    def __getitem__(self, index):
        img = default_loader(os.path.join(self.root, self.part_paths[index]), self.transform)
        return img, int(self.labels[index])

    def __len__(self):
        return self.length

    def getName(self):
        pass


def getloader(dataset, trainbatchsize, catenum):
    transformations = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='/home/wuxiaodong/data', train=True, download=False, transform=transformations)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainbatchsize, shuffle=False, num_workers=4)
        labels = []
        for batch_idx, (data, target) in enumerate(trainloader):
            labels.extend(target)
        if catenum !=0:
            casampler = sampler.categoryRandomSampler(numBatchCategory=catenum, targets=labels, batch_size=200)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainbatchsize, shuffle=False, num_workers=4,
                                                      sampler=casampler)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainbatchsize, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR100(root='/home/wuxiaodong/data', train=False, download=False, transform=transformations)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    if dataset  == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='/home/wuxiaodong/data', train=True, download=False, transform=transformations)
        testset = torchvision.datasets.CIFAR10(root='/home/wuxiaodong/data', train=False, download=False, transform=transformations)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainbatchsize, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    if dataset == "imagenet100":
        train_path = '/home/hechen/ILSVRC/ILSVRC2012_img_train/'
        val_path = '/home/hechen/ILSVRC/ILSVRC2012_img_val/'

        train_data = imagenet(train_path, train='./data/imagenet100/train_imgs.txt', transform=transformations)
        val_data = imagenet(val_path, train='./data/imagenet100/val_imgs.txt', transform=transformations)
        if catenum != 0:
            trainloader = torch.utils.data.DataLoader(train_data, batch_size=trainbatchsize, shuffle=False, 
                                                       sampler=sampler2.categoryRandomSampler(catenum, trainbatchsize),
                                                       num_workers=8, pin_memory=False)
        else:
            trainloader = torch.utils.data.DataLoader(train_data, batch_size=trainbatchsize, shuffle=True, 
                                                       num_workers=8, pin_memory=False)          
        testloader = torch.utils.data.DataLoader(val_data, batch_size=200, shuffle=False, sampler=None,
                                                   num_workers=8, pin_memory=False)
    return trainloader, testloader


def getdshloader(trainbatchsize):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    trainTransform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    trainset = torchvision.datasets.CIFAR10(root='/home/wuxiaodong/data', train=True, download=False,transform=trainTransform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainbatchsize, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='/home/wuxiaodong/data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    return trainloader, testloader

