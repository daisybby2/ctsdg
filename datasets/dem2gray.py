import numpy as np
from PIL import Image

def dem2gray(image):
    img_array=np.array(image)
    min=img_array.min()
    max=img_array.max()
    im_array = (img_array - min) / (max - min) * 255.0
    im_array = im_array.astype(np.float32)

    # 将numpy数组转换为PIL图像
    img = Image.fromarray(im_array)
    return img