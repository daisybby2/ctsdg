import os
from PIL import Image


IMG_EXTENSIONS = [#指定可以识别的图像格式，不用改了，有tif文件
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):#判断文件名是否是指定格式的图像

    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):#返回指定目录下所有符合条件的文件路径列表，max_dataset_size规定了最大数据个数

    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    images = sorted(images)

    return images[:min(max_dataset_size, len(images))]#满足条件的文件路径列表，字符串列表
