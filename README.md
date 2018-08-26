# DSH-Pytorch
This is a pytorch implementation of "Deep Supervised Hashing for Fast Image Retrieval".  
Note that tiis is a non-official implementation. If you want to compare with DSH in your research, please refer to the [code](https://github.com/lhmRyan/deep-supervised-hashing-DSH) here.  
This repo is just for personal usage and for communication. Some implementation details may be different.

# Requirement
Python 3.6.3  
Pytorch 0.3.1

# Training
You can conduct the training process using the following command  
```
python train.py  --gpu_id 0  --dataset cifar10 --codelength 48 --net dsh --lr 0.001
```
or 
```
python train.py  --gpu_id 0  --dataset cifar100 --codelen 48 --net resnet18 --numclass_perbatch 20 --lr 0.01 --postfix train1
```
* "gpu_id" is the ID of your GPU device
* "dataset" is a string to indictate the dataset you use. Now avaliable for "imagenet100", "cifar10", "cifar100"
* "codelength" is the number of hash codes
* "net" is the indicator of network backbone. "dsh"means the network adopted in by the original papaer. If you use "dsh", the dataset will be CIFAR-10 wtih input size 32\*32. "resnet18" is the default netwrok provided by Pytorch. If you use "resnet18", the inputsize will be 224\*224.
* "numclass_perbatch" is the number of classes in a mini-batch. We wil construct the mini-batch by sampling roughly equal number of images for each class. 0 means radomly sampling from the whole dataset.
* "lr" the initial learning rate. We don't set the learning rate decay strategy in this script. For "dsh", we train it from scratch. For "resnet18", we initialize it from model pretrained on ImageNet-1k and the learning rate of the last fully-connected layer is set to be 10 times of other layers. 

# Evaluation


<!-- # Evaluation
You can calculate MAP by running the following code:
```
python test.py --gpu_id 0 --dataset cifar100 --codelength 48 --model_path ./XX/snapshots/XX.pth
```
Note that  the quey set and the databae set are identical and they are both the validation set of the original dataset. -->


