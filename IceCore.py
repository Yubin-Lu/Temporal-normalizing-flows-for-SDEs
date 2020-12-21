# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:25:03 2020

@author: Jabin
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()

#from diffusioncharacterization.ctrw.random_walks import advection_diffusion_random_walk
from temporal_normalizing_flows.neural_flow import neural_flow
from temporal_normalizing_flows.latent_distributions import gaussian
from temporal_normalizing_flows.preprocessing import prepare_data

try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for GPU
except:
    pass


# loading IceCore_Oxygen18.xlsx
def LoadingData():
    data = pd.read_excel("IceCore_Oxygen18.xlsx")
    M = len(data)  # rows
    N = data.columns.size # columns 
    years = np.asarray(data.iloc[2:M,0:1].values,'float32')   
    O18 = np.asarray(data.iloc[2:M,1:2].values,'float32')   
    # years_Normal = years / np.sum(years)  
    O18_Normal = 100*(O18 - np.mean(O18)) / np.mean(O18)
    years_Normal = years
    L = len(years)
    l = 10
    sample_num = int(L/l)
    time = np.zeros(l)
    position = np. zeros([l, sample_num])
    for i in range(l):
        time[i] = np.mean(years_Normal[sample_num*i:sample_num*(i+1)-1])
        position[i,:] = O18_Normal[sample_num*i:sample_num*(i+1)].squeeze()
    return time, position



time, position = LoadingData()

#%% Time-dependent neural flow
x_sample = np.linspace(-10, 10, 1000)
t_sample = time
dataset = prepare_data(position, time, x_sample, t_sample)
flow = neural_flow(gaussian)
flow.train(dataset, 3000)

px, pz, jacob, z = flow.sample(dataset)


N = len(time);
for frame in range(N):
    # if frame%2 == 0:
    sns.distplot(position[frame, :], bins='auto',label="KDE")
    plt.title('t={}'.format(time[frame]))
    plt.plot(x_sample, px[frame,:],'r',label="TNF")
    plt.legend(loc=0,ncol=1)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.show()
