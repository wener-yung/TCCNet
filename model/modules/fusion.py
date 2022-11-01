import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

class Fusion(nn.Module):
    def __init__(self, channel, outchannel=32):
        super(Fusion, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        self.conv_upsample1 = conv(channel, channel, 3)
        self.conv_upsample2 = conv(channel, channel, 3)
        self.conv_upsample3 = conv(channel, channel, 3)
        self.conv_upsample4 = conv(channel, channel, 3)
        self.conv_upsample5 = conv(2 * channel, 2 * channel, 3)

        self.conv_concat2 = conv(2 * channel, 2 * channel, 3)
        self.conv_concat3 = conv(3 * channel, 3 * channel, 3)
        self.conv4 = conv(3 * channel, 3 * channel, 3)
        self.conv5 = conv(3 * channel, outchannel, 3)

    def forward(self, f1, f2, f3):
        f1x2 = self.upsample(f1, f2.shape[-2:])
        f1x4 = self.upsample(f1, f3.shape[-2:])
        f2x2 = self.upsample(f2, f3.shape[-2:])

        f2_1 = self.conv_upsample1(f1x2) * f2
        f3_1 = self.conv_upsample2(f1x4) * self.conv_upsample3(f2x2) * f3

        f1_2 = self.conv_upsample4(f1x2)
        f2_2 = torch.cat([f2_1, f1_2], 1)
        f2_2 = self.conv_concat2(f2_2)

        f2_2x2 = self.upsample(f2_2, f3.shape[-2:])
        f2_2x2 = self.conv_upsample5(f2_2x2)

        f3_2 = torch.cat([f3_1, f2_2x2], 1)
        f3_2 = self.conv_concat3(f3_2)

        f3_2 = F.relu(self.conv4(f3_2))
        out = self.conv5(f3_2)

        return out