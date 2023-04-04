import torch
import torch.nn as nn

import torchvision


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained=True)

        ##把VGG16的输入由3通道修改成单通道
        weight = vgg16.features[0].weight.sum(dim=1, keepdim=True)
        bias = vgg16.features[0].bias
        print(weight.shape)
        vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vgg16.features[0].weight = nn.Parameter(weight)
        vgg16.features[0].bias = nn.Parameter(bias)
        print(vgg16.features[0].weight.requires_grad)
        print(vgg16.features[0].bias.requires_grad)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, images):
        results = [images]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
