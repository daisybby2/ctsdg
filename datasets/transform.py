from PIL import Image

from torchvision import transforms


def image_transforms(load_size):#对图像预处理，重采样、转换成张量和归一化。如果是我们的DEM图像，应该还需要加入像素值映射到[0,255]的功能

    return transforms.Compose([
        # transforms.CenterCrop(size=(178, 178)),  # for CelebA
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR), #双线性插值，把图像缩放到load_size大小
        transforms.ToTensor(),#把图像转换成pytorch张量
        transforms.Normalize((0.5,  ), (0.5, ))#归一化，三通道，需要修改
    ])


def mask_transforms(load_size):#mask图像的预处理，重采样和转换成张量

    return transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
