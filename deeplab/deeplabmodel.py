
import torch
# from .model import MainModel as model
import torch
import torchvision
import os
import shutil
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import time
import glob
from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2lab, lab2rgb
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
from torchvision import models
from deeplab import utils
# from utils import GANLoss, init_model
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ResNet_50(nn.Module):
    def __init__(self, in_channels=1):
        super(ResNet_50, self).__init__()

        # Load the pre-trained ResNet-50 model
        self.resnet_50 = models.resnet50(pretrained=True)

        # Modify the first convolutional layer to accept 1-channel input
        self.resnet_50.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Use the layers up to the final layer before the fully connected layer
        self.resnet_50 = nn.Sequential(*list(self.resnet_50.children())[:-2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.resnet_50(x)
        return x

class ASSP(nn.Module):
    def __init__(self, in_channels, out_channels=256, final_out_channels=2):
        super(ASSP, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolutions with different dilation rates
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=9, dilation=9, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution after global average pooling
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

        # Final 1x1 convolution to combine features
        self.convf = nn.Conv2d(out_channels * 5, final_out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bnf = nn.BatchNorm2d(final_out_channels)

        # Global average pooling
        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 1x1 convolution
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        # 3x3 convolution with dilation 6
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        # 3x3 convolution with dilation 12
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        # 3x3 convolution with dilation 18
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)

        # Global average pooling, 1x1 convolution, and upsample
        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear', align_corners=True)

        # Concatenate all feature maps
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # Final 1x1 convolution
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)

        return x
    
class deeplabv3_encoder_decoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=2):
        super(deeplabv3_encoder_decoder, self).__init__()
        self.resnet = ResNet_50(in_channels=input_channels)
        self.aspp = ASSP(in_channels=2048, final_out_channels=1024)

        # Decoder layers
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)
        # )
    
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.BatchNorm2d(32),
                # nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.Sigmoid()  # Assuming the input images are normalized between 0 and 1
            )

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.resnet(x)  # Output should be [batch_size, 2048, H/32, W/32]
        x = self.aspp(x)
        # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)  # Upsample
        # print(x.shape)
        x = self.decoder(x)  # Decode
        return x
    
    
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [
            self.get_layers(
                num_filters * 2**i,
                num_filters * 2 ** (i + 1),
                s=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        model += [
            self.get_layers(num_filters * 2**n_down, 1, s=1, norm=False, act=False)
        ]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(
        self, ni, nf, k=4, s=2, p=1, norm=True, act=True
    ):  # when needing to make some repeatitive blocks of layers,
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)
        ]  # it's always helpful to make a separate method for that purpose
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MainModel(nn.Module):
    def __init__(
        self, net_G=None, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.0
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = utils.init_model(
                deeplabv3_encoder_decoder(),
                self.device
            )
        else:
            self.net_G = net_G.to(self.device)

        self.net_D = utils.init_model(
            PatchDiscriminator(input_c=3, num_filters=64, n_down=3),
            self.device
        )

        self.GANcriterion = utils.GANLoss(gan_mode="vanilla").to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data["L"].to(self.device)
        self.ab = data["ab"].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()