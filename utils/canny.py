import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def image_to_edge(image, sigma):#使用canny提取图像边。需要修改

    #gray_image = rgb2gray(np.array(tensor_to_image()(image)))##如果已经输入的灰度图，可以直接把这一句删除掉
    edge = image_to_tensor()(Image.fromarray(canny(image, sigma=sigma)))
    gray_image = image_to_tensor()(Image.fromarray(image))

    return edge, gray_image#返回canny提取的边和灰度图

