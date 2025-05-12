import random
import torch
from torch import nn
from model.stylegan.model import ConvLayer, PixelNorm, EqualLinear, Generator

class UNet(nn.Module):
    def __init__(self, channels, stride_list):
        super(UNet, self).__init__()
        # initial
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        # conv
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=stride_list[i+1], padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True),
                )
            )
        
        # deconv
        for i in reversed(range(len(channels) - 1)):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i+1], channels[i], kernel_size=3, stride=stride_list[i+1], padding=1, output_padding=1),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU(inplace=True),
                )
            )

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True),
        )

        #output
        self.output_layer = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        skip_connections = []
        
        # conv
        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)
        
        output = []
        
        # bottleneck
        x = self.bottleneck(x)

        output.append(x)

        skip_connections2 = skip_connections[:-1]
        
        # deconv
        for block, skip in zip(self.up_blocks, reversed(skip_connections2)):
            x = x + skip  # 
            x = block(x)
            output.append(x)
            print(x.shape)
        
        return output