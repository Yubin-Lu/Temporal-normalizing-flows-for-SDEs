# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:51:38 2020

@author: Yubin Lu
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
        self.t3 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.s3 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.t4 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.s4 = base_network(1 + dim // 2, dim // 2, hidden_dim)

    
    
    def forward(self, x):            
        lower, upper = x[:,0::2], x[:,1].reshape(-1, 1)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        lower_new = lower
        upper_new = t1_transformed + upper * torch.exp(s1_transformed)
        z1 = torch.cat([lower_new[:,0].reshape(-1, 1), upper_new, lower_new[:,1].reshape(-1, 1)], dim=1)
       
        lower1, upper1 = z1[:,0].reshape(-1, 1), z1[:,1:3]
        t2_transformed = self.t2(upper1)
        s2_transformed = self.s2(upper1)
        lower_new1 = t2_transformed + lower1 * torch.exp(s2_transformed)
        upper_new1 = upper1
        z2 = torch.cat([lower_new1, upper_new1], dim=1)
        
        lower2, upper2 = z2[:,0::2], z2[:,1].reshape(-1, 1)
        t3_transformed = self.t3(lower2)
        s3_transformed = self.s3(lower2)
        lower_new2 = lower2
        upper_new2 = t3_transformed + upper2 * torch.exp(s3_transformed)
        z3 = torch.cat([lower_new2[:,0].reshape(-1, 1), upper_new2, lower_new2[:,1].reshape(-1, 1)], dim=1)
       
        lower3, upper3 = z3[:,0].reshape(-1, 1), z3[:,1:3]
        t4_transformed = self.t4(upper3)
        s4_transformed = self.s4(upper3)
        lower_new3 = t4_transformed + lower3 * torch.exp(s4_transformed)
        upper_new3 = upper3
        z = torch.cat([lower_new3, upper_new3[:,0].reshape(-1, 1)], dim=1)
        log_det = torch.sum(s1_transformed + s2_transformed + s3_transformed + s4_transformed, dim=1)
        c1 = 1/(2*3.14159265)
        c2 = -torch.sum(z**2, 1)/2
        pz = torch.mul(c1, torch.exp(c2))
        log_px = torch.log(pz) + log_det
        px = torch.exp(log_px)
        return z, log_det, px

    
    def inverse(self, z):
        lower3, upper3 = z[:,0].reshape(-1, 1), z[:,1:3]
        t4_transformed = self.t4(upper3)
        s4_transformed = self.s4(upper3)
        lower_new3 = (lower3 - t4_transformed) * torch.exp(-s4_transformed)
        upper_new3 = upper3 
        x3 = torch.cat([lower_new3[:,0].reshape(-1, 1), upper_new3], dim=1)
        
        lower2, upper2 = x3[:,0::2], x3[:,1].reshape(-1, 1)
        t3_transformed = self.t3(lower2)
        s3_transformed = self.s3(lower2)
        lower_new2 = lower2
        upper_new2 = (upper2 - t3_transformed) * torch.exp(-s3_transformed)        
        x2 =  torch.cat([lower_new2[:,0].reshape(-1, 1), upper_new2, lower_new2[:,1].reshape(-1, 1)], dim=1)
        
        lower1, upper1 = x2[:,0].reshape(-1, 1), x2[:,1:3]
        t2_transformed = self.t2(upper1)
        s2_transformed = self.s2(upper1)
        lower_new1 = (lower1 - t2_transformed) * torch.exp(-s2_transformed)
        upper_new1 = upper1 
        x1 = torch.cat([lower_new1[:,0].reshape(-1, 1), upper_new1], dim=1)
        
        lower, upper = x1[:,0::2], x1[:,1].reshape(-1, 1)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        lower_new = lower
        upper_new = (upper - t1_transformed) * torch.exp(-s1_transformed)        
        x =  torch.cat([lower_new[:,0].reshape(-1, 1), upper_new], dim=1)
        log_det = torch.sum(-s1_transformed - s2_transformed - s3_transformed - s4_transformed, dim=1)
        return x, log_det
