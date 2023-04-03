import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.canny import image_to_edge
from datasets.transform import mask_transforms, image_transforms
from datasets.folder import make_dataset


class ImageDataset(Dataset):

    def __init__(self, image_root, mask_root, load_size, sigma=2., mode='test'):
        super(ImageDataset, self).__init__()

        self.image_files = make_dataset(dir=image_root)#获取合法image路径列表
        self.mask_files = make_dataset(dir=mask_root)#获取合法mask路径列表

        self.number_image = len(self.image_files)#获取图像个数
        self.number_mask = len(self.mask_files)#mask个数

        self.sigma = sigma#用于提取边缘的sigma
        self.mode = mode

        self.load_size = load_size
 
        self.image_files_transforms = image_transforms(load_size)   
        self.mask_files_transforms = mask_transforms(load_size)#image和mask的预处理

    def __getitem__(self, index):

        image = Image.open(self.image_files[index % self.number_image])
        image = self.image_files_transforms(image.convert('RGB'))#将读取的图像转换成RGB，应该是需要修改的，DEM是单通道图像。这一句是数据预处理

        if self.mode == 'train':#训练模式下
            mask = Image.open(self.mask_files[random.randint(0, self.number_mask - 1)])#随机选择一张mask掩膜
        else:#val验证模式或者test测试
            mask = Image.open(self.mask_files[index % self.number_mask])

        mask = self.mask_files_transforms(mask)#mask掩膜预处理
        ##
        threshold = 0.5
        ones = mask >= threshold
        zeros = mask < threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)
        ##生成二进制掩膜
        mask = 1 - mask#掩膜反转

        edge, gray_image = image_to_edge(image, sigma=self.sigma)#canny提取图像的边

        return image, mask, edge, gray_image#返回图像，掩膜，边图和灰度图

    def __len__(self):

        return self.number_image


def create_image_dataset(opts):

    image_dataset = ImageDataset(
        opts.image_root,
        opts.mask_root,
        opts.load_size,
        opts.sigma,
        opts.mode
    )

    return image_dataset#返回的是pytorch的dataset子类
