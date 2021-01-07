# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 20:23:49 2020

@author: Administrator
"""
import numpy as np
import itertools
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

from nf.TNF import *
from nf.models import NormalizingFlowModel


def StableVariable(m, alpha):
     V = np.pi/2 * (2*np.random.rand(m)-1)
     W = np.random.exponential(scale=1, size=m)
     y = np.sin(alpha * V) / (np.cos(V)**(1/alpha) ) * (np.cos( V*(1-alpha)) / W )**((1-alpha)/alpha)
     return y

def GeneratingData(T, dt):
    t = np.arange(0, T, dt)
    Nt = len(t)
    X0 = np.random.randn(500,2)
    x0 = X0[:,0:1]
    y0 = X0[:,1:]
    N = len(x0)
    alpha = 2.
    x = np.zeros((Nt, N))
    y = np.zeros((Nt, N))
    x[0, :] = x0.squeeze()
    y[0, :] = y0.squeeze()
    for i in range(Nt-1):
        # Ut = dt**(1/alpha) * StableVariable(N, alpha)
        # Vt = dt**(1/alpha) * StableVariable(N, alpha)
        Ut = dt**(1/2) * np.random.randn(N)
        Vt = dt**(1/2) * np.random.randn(N)
        # x[i+1, :] = x[i, :] - x[i, :]*y[i, :]*dt + 1*Ut
        # y[i+1, :] = y[i, :] + (4*y[i, :] - 1*y[i, :]**3)*dt + 2*Vt
        x[i+1, :] = x[i, :] + 1*(4*x[i, :] - 1*x[i, :]**3)*dt + 1*Ut
        y[i+1, :] = y[i, :] - x[i, :]*y[i, :]*dt + 2*Vt


    return t, x, y



def plot_data(x, **kwargs):
    plt.scatter(x[:,0], x[:,1], marker="x", **kwargs)
    # plt.xlim((-3, 3))
    # plt.ylim((-3, 3))


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=1, type=int) #复合两次RealNVP
    argparser.add_argument("--flow", default="RealNVP", type=str)
    argparser.add_argument("--iterations", default=10000, type=int)
    argparser.add_argument("--use-mixture", action="store_true")
    argparser.add_argument("--convolve", action="store_true")
    argparser.add_argument("--actnorm", action="store_true")
    args = argparser.parse_args() 
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    flow = eval(args.flow)
    flows = [flow(dim=2) for _ in range(args.flows)]
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = NormalizingFlowModel(prior, flows)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    T = 1
    dt = 0.05
    time, position_x, position_y = GeneratingData(T, dt)
    
    
    
    t = np.repeat(time, position_x.shape[1]).reshape(-1, 1)
    P_x = np.reshape(position_x, position_x.size, order='C').reshape(-1, 1)
    P_y = np.reshape(position_y, position_y.size, order='C').reshape(-1, 1)
    x = torch.Tensor(np.concatenate((P_x,P_y,t),axis=1))
    

    for i in range(args.iterations):
        optimizer.zero_grad()
        z, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.mean(prior_logprob + log_det)
        loss.backward()
        optimizer.step()
    xx = np.concatenate((position_x[19:20,:].T,position_y[19:20,:].T) ,axis=1)
    plt.subplot(1, 2, 1)
    plot_data(xx, color="black", alpha=0.5)
    plt.title("Training data")
    plt.subplot(1, 2, 2)
    samples = model.sample(500, t=0.95).data
    plot_data(samples, color="black", alpha=0.5)
    plt.title("Generated samples")
    plt.show()





