import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MyRefPad1d(nn.Module):
    def __init__(self, value):
        super(MyRefPad1d, self).__init__()
        self.value = value

    def forward(self, x):
        input_size = x.size()
        # input_size = [1,80,100]
        # ----------
        ref_pad1d = x.clone()
        for i in range(self.value):
            tmp1 = x[:, :, i + 1].unsqueeze(2)
            tmp2 = x[:, :, input_size[2] - i - 2].unsqueeze(2)
            ref_pad1d = torch.cat((tmp1, ref_pad1d), 2)
            ref_pad1d = torch.cat((ref_pad1d, tmp2), 2)

        return ref_pad1d


class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                # MyRefPad1d(3**i),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(3)
        ])

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            for i in range(3)
        ])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)
