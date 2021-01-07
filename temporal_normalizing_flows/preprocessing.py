import torch
import numpy as np
from collections import namedtuple

def prepare_data(particle_list, sample_list, particle_name_list=['t','x'], randomize=False):
    '''
    particle_list is a list of particles (time has to be the first dimension)
    sample_list is a list of samples (time has to be the first dimension)
    particle_name_list is the name of the particles for e.g. : ['t','x']
    '''

    grid_list = np.meshgrid(*sample_list, indexing='ij')
    rand_idx = np.random.permutation(grid_list[0].size)

    temp = grid_list[0].reshape(-1,1)
    for i in range(1,len(grid_list)):
        temp = np.concatenate((temp,grid_list[i].reshape(-1,1)),axis=1)

    grid_data = torch.tensor(temp, dtype=torch.float32, requires_grad=True)
    grid_dims = grid_list[0].shape

    tuple_names = ['grid_data', 'grid_dims', 'unrand_idx']
    particle_name_list.reverse()
    for i in range(len(grid_list)):
        tuple_names.insert(0,'particle_'+particle_name_list[i])

    tensor_list = []
    for i in range(1,len(particle_list)):
        tensor_list.append(torch.tensor(particle_list[i], dtype=torch.float32))

    tensor_list.insert(0,torch.tensor(np.ones_like(particle_list[1])*particle_list[0][:, None], dtype=torch.float32))
    
    unrand_idx = np.empty(rand_idx.size, rand_idx.dtype)
    unrand_idx[rand_idx] = np.arange(rand_idx.size) # used to unrandomize the data

    neural_flow_data = namedtuple('neural_flow_data', tuple_names)
    dataset = neural_flow_data(*tensor_list, grid_data, grid_dims, unrand_idx)

    return dataset

if __name__ == '__main__':
    import os, sys
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    from TNFforDWwithBM import GeneratingData
    

    #%% Time-dependent neural flow
    T = 1
    dt = 0.05
    time, position = GeneratingData(T, dt) 
    x_sample = np.linspace(-10, 10, 1000)
    t_sample = time

    d1 = prepare_data([time, position], [t_sample, x_sample])
