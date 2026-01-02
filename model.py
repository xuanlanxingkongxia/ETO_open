import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, channels: list, do_bn):
        super().__init__()
        self.linears = []
        self.do_bn = do_bn
        self.bns = []
        for i in range(len(channels) - 1):
            self.linears.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, bias=True))
            self.bns.append(nn.BatchNorm1d(channels[i]))

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if self.do_bn:
                x = self.bns[i](x)
            x = F.relu(x)
        return x

