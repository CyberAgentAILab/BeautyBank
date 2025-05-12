import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn


def Conv4x4(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.1)
    )

class DeConv4x4(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.BN = nn.BatchNorm2d(c_out)
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, input, skip):
        x = self.deconv(input)
        x = self.BN(x)
        x = self.LReLU(x)

        #h = skip
        out = torch.cat((x, skip), dim=1)
        return out


class AADLayer(nn.Module):
    def __init__(self, c_in, c_id, c_att):
        super().__init__()
        self.c_in = c_in
        self.c_id = c_id
        self.c_att = c_att

        self.conv1 = nn.Conv2d(c_att, c_in, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(c_att, c_in, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(c_id, c_in)
        self.fc2 = nn.Linear(c_id, c_in)

        self.norm = nn.InstanceNorm2d(c_in, affine=False)

        self.conv_h = nn.Conv2d(c_in, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, h_in, z_id, z_att):
        h_bar = self.norm(h_in)

        gamma_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)

        A = gamma_att * h_bar + beta_att

        gamma_id = self.fc1(z_id).view(z_id.size(0),self.c_in, 1, 1).expand_as(h_bar)
        beta_id = self.fc2(z_id).view(z_id.size(0), self.c_in, 1, 1).expand_as(h_bar)

        I = gamma_id * h_bar + beta_id

        M = torch.sigmoid(self.conv_h(h_bar))

        h_out = (torch.ones_like(M).to(M.device) - M) * A + M * I

        return h_out


class AADResBlk(nn.Module):
    def __init__(self, c_in, c_out, c_id, c_att):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.c_id = c_id
        self.c_att = c_att

        self.AAD1 = AADLayer(c_in, c_id, c_att)
        self.relu1 = nn.ReLU(inplace=True)
        self.Conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1)

        self.AAD2 = AADLayer(c_in, c_id, c_att)
        self.relu2 = nn.ReLU(inplace=True)
        self.Conv2 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)

        if self.c_in!= self.c_out:
            self.AAD3 = AADLayer(c_in, c_id, c_att)
            self.relu3 = nn.ReLU(inplace=True)
            self.Conv3 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)

    def forward(self, h, z_id, z_att):
        x = self.AAD1(h, z_id, z_att)
        x = self.relu1(x)
        x = self.Conv1(x)

        x = self.AAD2(x, z_id, z_att)
        x = self.relu2(x)
        x = self.Conv2(x)

        if self.c_in != self.c_out:
            h = self.AAD3(h, z_id, z_att)
            h = self.relu3(h)
            h = self.Conv3(h)

        out = x + h

        return out





