# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:10:46 2020

@author: Jabin
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as scio

sns.set()

#from diffusioncharacterization.ctrw.random_walks import advection_diffusion_random_walk
from temporal_normalizing_flows.neural_flow import neural_flow
from temporal_normalizing_flows.latent_distributions import gaussian
from temporal_normalizing_flows.preprocessing import prepare_data

try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for GPU
except:
    pass


def GeneratingData(T, dt):
    t = np.arange(0, T, dt)
    Nt = len(t)
    X0 = np.random.randn(500)
    x0 = X0
    N = len(x0)
    x = np.zeros((Nt, N))
    x[0, :] = x0
    for i in range(Nt-1):
        Ut = dt**(1/2) * np.random.randn(N)
        #double-well systems with Brownian motion
        x[i+1, :] = x[i, :] + 1*(4*x[i, :] - 1*x[i, :]**3)*dt + 1*Ut      
    return t, x


dataFile = 'density10.mat' #come from Fokker-Planck equation
data = scio.loadmat(dataFile)
density = data['P']
T = 1
dt = 0.05
time, position = GeneratingData(T, dt) 

plt.figure(figsize=(8, 5))
plt.plot(time, position)

plt.xlabel('t')
plt.ylabel('x')
plt.xlim([0, T])
plt.show()

frame = 10
sns.distplot(position[frame, :], bins='auto')
plt.title('t={}'.format(time[frame]))
plt.show()

#%% Time-dependent neural flow
x_sample = np.linspace(-10, 10, 1000)
t_sample = time
dataset = prepare_data(position, time, x_sample, t_sample)
flow = neural_flow(gaussian)
flow.train(dataset, 10000)

px, pz, jacob, z = flow.sample(dataset)
plt.contourf(px)
plt.xlabel('x')
plt.ylabel('t')

N = len(time)
plt.figure(figsize=(8, 5))
for frame in range(N):
    if frame%5 == 0:
        sns.distplot(position[frame, :], bins='auto',label="KDE")
        plt.title('t={}'.format(time[frame]))
        plt.plot(x_sample, px[frame,:],'r',label="TNF")
        plt.plot(x_sample, density[frame,:],'g',label="True")
        plt.legend(loc=0,ncol=1)
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.show()
