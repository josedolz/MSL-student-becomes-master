import torch.nn.init as init
import torch.nn.functional as F
import pdb
import math

import torch
from torch import nn

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)

class UNet_Mixed(nn.Module):
    def __init__(self, num_classes):
        super(UNet_Mixed, self).__init__()
        self.enc1 = _EncoderBlock(1, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)

        ### Decoder 1 (Full-sup) ####
        self.dec4_A = _DecoderBlock(1024, 512, 256)
        self.dec3_A = _DecoderBlock(512, 256, 128)
        self.dec2_A = _DecoderBlock(256, 128, 64)
        self.dec1_A = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final_A = nn.Conv2d(64, num_classes, kernel_size=1)

        ### Decoder 2 (Weakly-sup) ####
        self.dec4_B= _DecoderBlock(1024, 512, 256)
        self.dec3_B = _DecoderBlock(512, 256, 128)
        self.dec2_B = _DecoderBlock(256, 128, 64)
        self.dec1_B = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final_B = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)

        ### Decoder 1 (Full-sup) ####
        dec4_A = self.dec4_A(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3_A = self.dec3_A(torch.cat([dec4_A, F.upsample(enc3, dec4_A.size()[2:], mode='bilinear')], 1))
        dec2_A = self.dec2_A(torch.cat([dec3_A, F.upsample(enc2, dec3_A.size()[2:], mode='bilinear')], 1))
        dec1_A = self.dec1_A(torch.cat([dec2_A, F.upsample(enc1, dec2_A.size()[2:], mode='bilinear')], 1))
        final_A = self.final_A(dec1_A)

        ### Decoder 2 (Weakly-sup) ####
        dec4_B = self.dec4_B(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3_B = self.dec3_B(torch.cat([dec4_B, F.upsample(enc3, dec4_B.size()[2:], mode='bilinear')], 1))
        dec2_B = self.dec2_B(torch.cat([dec3_B, F.upsample(enc2, dec3_B.size()[2:], mode='bilinear')], 1))
        dec1_B = self.dec1_B(torch.cat([dec2_B, F.upsample(enc1, dec2_B.size()[2:], mode='bilinear')], 1))
        final_B = self.final_B(dec1_B)

        return [F.upsample(final_A, x.size()[2:], mode='bilinear'),
                F.upsample(final_B, x.size()[2:], mode='bilinear')]
