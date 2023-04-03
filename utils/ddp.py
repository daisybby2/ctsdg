import torch
import torch.nn as nn
from torch.utils import data


def data_sampler(dataset, shuffle, distributed):#输入dataset对象，根据shuffle和distributed决定创建随机采样器还是分布式采样器
    
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)#返回采样器


