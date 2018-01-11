# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:53:25 2017

网络数据

@author: WangDian
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 容量计算函数
def Cn_t_T(T):
    Cn_t = 0.004051*T**3 - 0.6669*T**2 + 38.82*T + 2557
    return Cn_t

# 滤波 传入一维参数
def Filter(a, N):
    x = np.zeros([len(a)])
    for i in range(N, len(a)):
        x[i] = np.median(a[i-N:i+1])
    x[0:N] = a[0:N]
    return x

#==============================================================================
# 定义网络
#==============================================================================
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.wb1   = nn.Linear(2, 50)
        self.wb2   = nn.Linear(50, 25)
        self.wb3   = nn.Linear(25, 10)
        self.wb4   = nn.Linear(10, 1)
    def forward(self, x):
        x = F.relu(self.wb1(x))
        x = F.relu(self.wb2(x))
        x = F.relu(self.wb3(x))
        x = self.wb4(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.wb1   = nn.Linear(3, 15)
        self.wb2   = nn.Linear(15, 10)
        self.wb3   = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.wb1(x))
        x = F.relu(self.wb2(x))
        x = self.wb3(x)
        return x

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.wb1   = nn.Linear(2, 10)
        self.wb2   = nn.Linear(10, 1)

    def forward(self, x):
        x = F.tanh(self.wb1(x))
        x = self.wb2(x)
        return x




