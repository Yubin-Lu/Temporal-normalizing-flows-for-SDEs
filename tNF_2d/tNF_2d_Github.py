# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:10:28 2021

@author: Jabin
"""
import numpy as np
import itertools
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio
import scipy.stats as st
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

from nf.TNF import *
from nf.models import NormalizingFlowModel


try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for GPU
except:
    pass



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True



def StableVariable(m, alpha):
     V = np.pi/2 * (2*np.random.rand(m)-1)
     W = np.random.exponential(scale=1, size=m)
     y = np.sin(alpha * V) / (np.cos(V)**(1/alpha) ) * (np.cos( V*(1-alpha)) / W )**((1-alpha)/alpha)
     return y

def GeneratingData(T, dt, n_samples):
    t = np.arange(0, T, dt)
    
    Nt = len(t)
    
    initial_state = 0.5*np.random.randn(n_samples,2)  # for Ex4, Ex5
    # initial_state = 1*np.random.randn(n_samples,2) # for Ex1, Ex3
    x0 = initial_state[:,0:1]
    y0 = initial_state[:,1:]
    N = n_samples
    alpha = 1.5
    x = np.zeros((Nt, N))
    y = np.zeros((Nt, N))
    x[0, :] = x0.squeeze()
    y[0, :] = y0.squeeze()
    
    

    for i in range(Nt-1):
        Ut = dt**(1/alpha) * StableVariable(N, alpha)
        Vt = dt**(1/alpha) * StableVariable(N, alpha)
        UUt = dt**(1/2) * np.random.randn(N)
        VVt = dt**(1/2) * np.random.randn(N)
        
        # ## Ex1
        # x[i+1, :] = x[i, :] + (4*x[i, :] - x[i, :]**3)*dt + x[i, :]*Ut
        # y[i+1, :] = y[i, :] - x[i, :]*y[i, :]*dt + y[i, :]*Vt 
        
        # ## Ex3
        # x[i+1, :] = x[i, :] + (8*x[i, :] - x[i, :]**3)*dt + UUt
        # y[i+1, :] = y[i, :] + (8*y[i, :] - y[i, :]**3)*dt + VVt
        
        # ## Ex4
        # x[i+1, :] = x[i, :] + (4*x[i, :] - x[i, :]**3)*dt + 1*UUt
        # y[i+1, :] = y[i, :] - x[i, :]*y[i, :]*dt + 1*VVt
        
        ## Ex5
        x[i+1, :] = x[i, :] + (x[i, :] - x[i, :]**3)*dt + Ut
        y[i+1, :] = y[i, :] + (y[i, :] - y[i, :]**3)*dt + Vt

        b=np.empty(0).astype(int)
        for j in range(n_samples):
            if (np.abs(x[:,j])>1e4).any() or (np.abs(y[:,j])>1e4).any():
                b = np.append(b,j)
        x1 = np.delete(x,b,axis=1)
        y1 = np.delete(y,b,axis=1)


    return t, x1, y1


def sample2density(x, u, v, du, dv):
    m, n = u.shape
    l, s =x.shape
    count = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            for k in range(l):
                if x[k,0]>=u[i,j]- du/2 and x[k,0]<u[i,j] + du/2 and x[k,1]>=v[i,j]- dv/2 and x[k,1]<v[i,j]+ dv/2:
                    count[i,j] += 1
    return count/(l*du*dv)
    
    
    
    
    


def plot_data(x, **kwargs):
    
    plt.scatter(x[:,0], x[:,1], s=1, marker="o", **kwargs)

    x_ticks = np.linspace(-8, 8, 5)
    y_ticks = np.linspace(-8, 8, 5)
    plt.xticks(x_ticks)  
    plt.yticks(y_ticks)
    plt.xlim((-8, 8))
    plt.ylim((-8, 8))
    plt.xlabel("$X_1$",fontsize=30)
    plt.ylabel("$X_2$",fontsize=30)
    plt.tick_params(labelsize=20)


if __name__ == "__main__":

    setup_seed(123)
    argparser = ArgumentParser()
    argparser.add_argument("--n", default=500, type=int)
    argparser.add_argument("--flows", default=2, type=int)
    argparser.add_argument("--flow", default="RealNVP_2d", type=str)
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--dim", default=2, type=int)
    args = argparser.parse_args()



    flow = eval(args.flow)
    flows = [flow(dim=2) for _ in range(args.flows)]
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = NormalizingFlowModel(prior, flows, args.dim)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    T = 1.05
    dt = 0.05
    sample_size = args.n
    time, position_x, position_y = GeneratingData(T, dt, args.n)
    
    ##data organization
    t = np.repeat(time, position_x.shape[1]).reshape(-1, 1)
    P_x = np.reshape(position_x, position_x.size, order='C').reshape(-1, 1)
    P_y = np.reshape(position_y, position_y.size, order='C').reshape(-1, 1)
    x = torch.Tensor(np.concatenate((P_x,P_y,t),axis=1))
    

    Loss = np.zeros([args.iterations, 1])
    for i in range(args.iterations):
        optimizer.zero_grad()
        z, prior_logprob, log_det, px = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.mean(prior_logprob + log_det)
        loss.backward()
        optimizer.step()
        Loss[i] = loss.cpu().detach().numpy()
        if i%100==0:
            print('iters:', i, 'Loss:', Loss[i])
    
    
    
    #resampling
    plt.figure(figsize=(36,16))
    plt.subplot(2, 4, 1)
    ## position_x[0:1,:] means P(x, 0) 
    xxx = np.concatenate((position_x[0:1,:].T, position_y[0:1,:].T), axis=1)
    plot_data(xxx, color="black", alpha=0.5)
    plt.title("$t=0.0$",fontsize=30)
    plt.subplot(2, 4, 2)
    ## the array position_x[5:6,:] means P(x, 0.25) because 5*dt=0.25, where dt=0.05
    xxx = np.concatenate((position_x[4:5,:].T, position_y[4:5,:].T), axis=1)
    plot_data(xxx, color="black", alpha=0.5)
    plt.title("$t=0.2$",fontsize=30)
    plt.subplot(2, 4, 3)
    xxx = np.concatenate((position_x[12:13,:].T, position_y[12:13,:].T), axis=1)
    plot_data(xxx, color="black", alpha=0.5)
    plt.title("$t=0.6$",fontsize=30)
    plt.subplot(2, 4, 4)
    xxx = np.concatenate((position_x[20:21,:].T, position_y[20:21,:].T), axis=1)
    plot_data(xxx, color="black", alpha=0.5)
    plt.title("$t=1.0$",fontsize=30)
    plt.subplot(2, 4, 5)
    samples = model.sample(sample_size, t=0).cpu().detach().numpy()
    plot_data(samples, color="black", alpha=0.5)
    plt.subplot(2, 4, 6)
    samples = model.sample(sample_size, t=0.2).cpu().detach().numpy()
    plot_data(samples, color="black", alpha=0.5)
    plt.subplot(2, 4, 7)
    samples = model.sample(sample_size, t=0.6).cpu().detach().numpy()
    plot_data(samples, color="black", alpha=0.5)
    plt.subplot(2, 4, 8)
    samples = model.sample(sample_size, t=1.0).cpu().detach().numpy()
    plot_data(samples, color="black", alpha=0.5)
    plt.show()



    
    
    ## Loss function
    # np.save("Ex4_Loss.npy", Loss)
    q=np.arange(0,args.iterations)
    plt.plot(q[:],Loss[:],'r')
    plt.show()
    

    

    


    x1, x2 = -9.8, 9.8
    y1, y2 = -9.8, 9.8
    grid_size = 98
    du = (x2-x1) / grid_size
    dv = (y2-y1) / grid_size
    u, v = np.meshgrid(np.linspace(x1, x2, grid_size+1), np.linspace(y1, y2, grid_size+1))
    u1 = np.reshape(u, u.size, order='C').reshape(-1, 1)
    v1 = np.reshape(v, v.size, order='C').reshape(-1, 1)
    uu = torch.Tensor(np.concatenate((u1,v1),axis=1))
    px1_estimated = model.sample_px(uu, t=0.)
    px1 = px1_estimated.cpu().detach().reshape(u.shape).numpy()     # Estimated density
    px2_estimated = model.sample_px(uu, t=0.2)
    px2 = px2_estimated.cpu().detach().reshape(u.shape).numpy()     # Estimated density
    px3_estimated = model.sample_px(uu, t=0.6)
    px3 = px3_estimated.cpu().detach().reshape(u.shape).numpy()     # Estimated density
    px4_estimated = model.sample_px(uu, t=1.0)
    px4 = px4_estimated.cpu().detach().reshape(u.shape).numpy()     # Estimated density
    
    

        

    

    
    


    

