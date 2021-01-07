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
from temporal_normalizing_flows.realnvp import realnvp

if torch.cuda.is_available():
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


if __name__ == '__main__':

    dataFile = 'density10.mat' #come from Fokker-Planck equation
    data = scio.loadmat(dataFile)
    density = data['P']
    T = 1
    dt = 0.05
    time, position = GeneratingData(T, dt) 

    # plt.figure(figsize=(8, 5))
    # plt.plot(time, position)

    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.xlim([0, T])
    # plt.show()

    # frame = 10
    # sns.distplot(position[frame, :], bins='auto')
    # plt.title('t={}'.format(time[frame]))
    # plt.show()

    #%% Time-dependent neural flow
    x_sample = np.linspace(-10, 10, 1000)
    t_sample = time
    num_grid_points = x_sample.shape[0]*t_sample.shape[0]

    dataset = prepare_data([time, position], [t_sample, x_sample], ['t','x'])
    flow = realnvp(gaussian,1,2, num_grid_points, num_coupling=3, perturb=True)

    # # Testing the forward (inference) and inverse operations of realnvp
    # # The two tensors printed should be identical for invertibility
    # log_px_1, log_pz_1, detjacob_1, z_1 = flow.inference(dataset)
    # x_ = flow.inverse(z_1)
    # print(dataset.grid_data[80:83])
    # print(x_[80:83])
    # exit()

    flow.train(dataset, 500)

    px, pz, detjacob, z = flow.sample(dataset)
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
