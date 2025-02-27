import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

from models.discriminator.discriminator import Discriminator
from models.generator.generator import Generator
from models.generator.vgg16 import VGG16FeatureExtractor
from options.train_options import TrainOptions
from datasets.dataset import create_image_dataset
from utils.distributed import synchronize
from utils.ddp import data_sampler
from trainer import train


opts = TrainOptions().parse

os.makedirs('{:s}'.format(opts.save_dir), exist_ok=True)#配置文件的保存路径吧

is_cuda = torch.cuda.is_available()#检测是否有可用的GPU
if is_cuda:

    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1#获取可用的GPU数量
    print('GPU number: ', n_gpu)
    opts.distributed = n_gpu > 1
    if opts.distributed:#如果GPU数量>1，则分布式训练，否则单卡训练，这是分布训练的设置
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

# model & load model
generator = Generator(image_in_channels=1, edge_in_channels=2, out_channels=1)#可以修改图像的通道数
discriminator = Discriminator(image_in_channels=1, edge_in_channels=2)
extractor = VGG16FeatureExtractor()

# cuda
if is_cuda:
    generator, discriminator, extractor = generator.cuda(), discriminator.cuda(), extractor.cuda()

# optimizer
if opts.finetune == True:
    print('Fine tune...')
    lr = opts.lr_finetune
    generator.freeze_ec_bn = True
else:
    lr = opts.gen_lr

generator_optim = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr * opts.D2G_lr)

# load checkpoints
if opts.pre_trained != '':
    ckpt_dict = torch.load(opts.pre_trained, map_location=lambda storage, loc: storage)
    opts.start_iter = ckpt_dict['n_iter']
    generator.load_state_dict(ckpt_dict['generator'])
    discriminator.load_state_dict(ckpt_dict['discriminator'])

    print('Starting from iter', opts.start_iter)
else:
    print('Starting from iter', opts.start_iter)


if opts.distributed:

    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
    )
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
    )

# dataset
image_dataset = create_image_dataset(opts)#创建一个image训练集，是pytorch的dataset类
print(image_dataset.__len__())#打印训练集长度

image_data_loader = data.DataLoader(#加载数据集
    image_dataset,
    batch_size=opts.batch_size,#batch_size默认是6
    sampler=data_sampler(#为数据集设置一个采样器
        image_dataset, shuffle=True, distributed=opts.distributed
    ),
    drop_last=True#最后的数据不足一个batch则丢弃
)

# training
train(opts, image_data_loader, generator, discriminator, extractor, generator_optim, discriminator_optim, is_cuda)#进行网络训练



