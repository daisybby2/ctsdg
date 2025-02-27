import torch
import torch.nn as nn


class Bottleneck(nn.Module):#一个残差块，输入x，x经过一次卷积->BN操作->relu，再卷积->BN操作，直接+x实现跳跃连接，再relu

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual#实现跳跃连接
        out = self.relu(out)

        return out


class Feature2Structure(nn.Module):

    def __init__(self, inplanes=64, planes=16):
        super(Feature2Structure, self).__init__()

        self.structure_resolver = Bottleneck(inplanes, planes)#残差块，将高维纹理特征转换成低维特征
        self.out_layer = nn.Sequential(#容器，将残差块的输出映射成二值边图
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, structure_feature):

        x = self.structure_resolver(structure_feature)
        structure = self.out_layer(x)
        return structure


class Feature2Texture(nn.Module):

    def __init__(self, inplanes=64, planes=16):
        super(Feature2Texture, self).__init__()

        self.texture_resolver = Bottleneck(inplanes, planes)
        self.out_layer = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Tanh()
        )

    def forward(self, texture_feature):

        x = self.texture_resolver(texture_feature)
        texture = self.out_layer(x)
        return texture