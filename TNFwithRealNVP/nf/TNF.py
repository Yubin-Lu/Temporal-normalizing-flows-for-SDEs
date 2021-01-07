# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:51:38 2020

@author: Administrator
"""
import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class RealNVP(nn.Module):
    """
    Non-volume preserving flow.

    [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim = 32, base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.s1 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.t2 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.s2 = base_network(1 + dim // 2, dim // 2, hidden_dim)

    def forward(self, x):
        lower, upper = x[:,0::2], x[:,1].reshape(-1, 1)
        # print(lower.shape)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        lower_new = lower
        upper_new = t1_transformed + upper * torch.exp(s1_transformed)
        z1 = torch.cat([lower_new[:,0].reshape(-1, 1), upper_new], dim=1)
        # z1 = torch.cat([lower_new[:,0].reshape(-1, 1), upper_new, lower_new[:,1].reshape(-1, 1)], dim=1)
        # lower1, upper1 = z1[:,0].reshape(-1, 1), z1[:,1:3]
        # # print(lower.shape)
        # t2_transformed = self.t2(upper1)
        # s2_transformed = self.s2(upper1)
        # lower_new1 = t2_transformed + lower1 * torch.exp(s2_transformed)
        # upper_new1 = upper1
        # z = torch.cat([lower_new1, upper_new1[:,0].reshape(-1, 1)], dim=1)
        # # print(z.shape)
        log_det = torch.sum(s1_transformed, dim=1)
        return z1, log_det

    def inverse(self, z):
        lower, upper = z[:,0::2], z[:,1].reshape(-1, 1)
        # print(lower.shape)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        lower_new = lower
        upper_new = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x1 = torch.cat([lower_new[:,0].reshape(-1, 1), upper_new], dim=1)
        # x1 = torch.cat([lower_new[:,0].reshape(-1, 1), upper_new, lower_new[:,1].reshape(-1, 1)], dim=1)
        # lower1, upper1 = x1[:,0].reshape(-1, 1), x1[:,1:3]
        # # print(lower.shape)
        # t2_transformed = self.t2(upper1)
        # s2_transformed = self.s2(upper1)
        # lower_new1 = (lower1 - t2_transformed) * torch.exp(-s2_transformed)
        # upper_new1 = upper1 
        # x = torch.cat([lower_new, upper_new1[:,0].reshape(-1, 1)], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1)
        return x1, log_det
