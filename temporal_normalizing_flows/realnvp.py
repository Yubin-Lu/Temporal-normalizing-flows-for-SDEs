import numpy as np
import torch
import torch.nn as nn

class realnvp(nn.Module):
    '''
    d_val -> number of dimensions to be preserved (equation 4 in https://arxiv.org/pdf/1605.08803.pdf)
    D_val -> Total number of dimensions in process
    Note: Component 1 of d_val is always time
    '''
    def __init__(self, latent_distribution, d_val, D_val, num_grid_points, num_coupling=1):
        super().__init__()
        self.d_val = d_val # First few "unperturbed dimensions"
        self.D_val = D_val # Total number of dimensions in the flow
        self.num_coupling = num_coupling 
        self.num_grid_points = num_grid_points

        self.s_nets, self.t_nets, self.z0 = self.initialize_networks()
        self.latent_dist = latent_distribution


    def initialize_networks(self):

        s_nets = nn.ModuleList()
        t_nets = nn.ModuleList()

        for i in range(self.num_coupling):
            # Builds the network and the initial condition parameter for the integration.
            s_nets.append(nn.Sequential(nn.Linear(self.d_val, 30), nn.Tanh(),
                                    nn.Linear(30, 30),  nn.Tanh(),
                                    nn.Linear(30, (self.D_val-self.d_val))))

            t_nets.append(nn.Sequential(nn.Linear(self.d_val, 30),  nn.Tanh(),
                                                nn.Linear(30, 30),  nn.Tanh(),
                                                nn.Linear(30, (self.D_val-self.d_val))))

        z0 = nn.Sequential(nn.Linear(1, 100), nn.Tanh(),
                           nn.Linear(100, 1))

        return s_nets, t_nets, z0

    def forward(self, dataset):
        x_ = dataset.grid_data

        for i in range(self.num_coupling):

            s_net = self.s_nets[i]
            t_net = self.t_nets[i]

            # Go from data to latent space (inference) to find z
            xd_ = x_[:,:self.d_val]
            xd_D = x_[:,self.d_val:]

            s_ = s_net(xd_)
            scaling = xd_D*torch.exp(s_)
            translation = t_net(xd_)

            zd_D = scaling + translation
            z_ = torch.stack((xd_,zd_D),axis=1)[:,:,0]

            # Need to find logdetjacob
            if i == 0:
                logdetjac = torch.sum(s_,dim=1,keepdim=True)
            else:
                logdetjac += torch.sum(s_,dim=1,keepdim=True)

            x_ = z_

        return x_, logdetjac

    def inference(self, dataset):

        z_, logdetjac = self.forward(dataset)

        # Operations to calculate total log likelihood
        log_pz = self.latent_dist.log_pz(z_[:,self.d_val:],z_[:,:self.d_val])
        log_px = log_pz + logdetjac

        return log_px, log_pz, logdetjac, z_


    def inverse(self,latent_space):
        z_ = latent_space

        inverse_transform_range = np.arange(self.num_coupling)[::-1]

        for i in inverse_transform_range:
            s_net = self.s_nets[i]
            t_net = self.t_nets[i]

            zd_ = z_[:,0:self.d_val]
            zd_D = z_[:,self.d_val:]

            xd_ = zd_

            t_term = t_net(zd_)
            s_term = s_net(zd_)

            xd_D = (zd_D - t_term)*torch.exp(-s_term)
            z_ = torch.stack((xd_, xd_D), dim=1)[:,:,0]

        return z_


    def sample_grid(self, results_regular_grid, dataset):

        # Samples location of particles from log_px grid.
        min_val_list = torch.min(dataset.grid_data, dim=0)[0]
        max_val_list = torch.max(dataset.grid_data, dim=0)[0]

        normalized_particle_list = []
        for i in range(len(min_val_list)):
            minval = min_val_list[i]
            maxval = max_val_list[i]
            if minval != maxval:
                normalized_particle_list.append((dataset[i] - minval)/(maxval-minval))
            else:
                normalized_particle_list.append(dataset[i])

        normalized_loc = torch.stack(tuple(normalized_particle_list),dim=-1)[None, :, :, :]
        high_D_results = results_regular_grid.reshape(dataset.grid_dims)[None, None, :, :]

        interpolated_data = torch.nn.functional.grid_sample(high_D_results, normalized_loc).squeeze()

        return interpolated_data

    def train(self, dataset, iterations):
        # trains normalizing flow
        optimizer = torch.optim.Adam(self.parameters())

        for it in np.arange(iterations):
            log_px_grid = self.inference(dataset)[0]

            # print(log_px_grid.shape)
            # print(dataset.grid_data.shape)
            log_px_samples = self.sample_grid(log_px_grid, dataset)
            loss = -torch.mean(log_px_samples)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if it % 1000 == 0:
            print(it, loss.item())

    def sample(self, dataset):
        log_px, log_pz, logdetjacob, z = self.inference(dataset)
        px = torch.exp(log_px).cpu().detach().reshape(dataset.grid_dims).numpy()
        pz = torch.exp(log_pz).cpu().detach().reshape(dataset.grid_dims).numpy()
        detjacob = torch.exp(logdetjacob).cpu().detach().reshape(dataset.grid_dims).numpy()
        z = z[:,self.d_val:].cpu().detach().reshape(dataset.grid_dims).numpy()

        return px, pz, detjacob, z
    
    def derivatives(self, jacob, z, dataset):
        f = jacob

        df = torch.autograd.grad(f, dataset.grid_data, torch.ones_like(f), create_graph=True)[0]
        f_t = df[:, 0:1]
        f_x = df[:, 1:2]
        f_xx = torch.autograd.grad(f_x, dataset.grid_data, torch.ones_like(f_x), retain_graph=True)[0][:, 1:2]

        z0 = self.z0(dataset.grid_data[:, 0:1])
        z0_t = torch.autograd.grad(z0, dataset.grid_data, torch.ones_like(z0), retain_graph=True)[0][:, 0:1]

        z_t = self.integrate(f_t, dataset, self.z0) - z0 + z0_t

        pz = self.latent_dist.pz(z, 0)
        pz_t = self.latent_dist.pz_t(z, 0)
        pz_z = self.latent_dist.pz_z(z, 0)
        pz_zz = self.latent_dist.pz_zz(z, 0)
        
        # Actually calculating the derivs
        px_t = (pz_t + pz_z * z_t) * f + pz * f_t
        #px_x = pz_z * f**2 + pz * f_x
        px_xx = pz_zz * f**3 + 3 * f * f_x * pz_z + pz * f_xx
        
        return px_t, px_xx
    
    def log_derivs(self, jacob, z, dataset):
        f = torch.log(jacob)

        df = torch.autograd.grad(f, dataset.grid_data, torch.ones_like(f), create_graph=True)[0]
        f_t = df[:, 0:1]
        f_x = df[:, 1:2]
        f_xx = torch.autograd.grad(f_x, dataset.grid_data, torch.ones_like(f_x), retain_graph=True)[0][:, 1:2]

        z0 = self.z0(dataset.grid_data[:, 0:1])
        z0_t = torch.autograd.grad(z0, dataset.grid_data, torch.ones_like(z0), retain_graph=True)[0][:, 0:1]

        z_t = self.integrate(f_t*torch.exp(f), dataset, 0) - z0 +z0_t
        
        log_px_t = -z * z_t + f_t
        log_px_x = -z * torch.exp(f) + f_x
        log_px_xx = f_xx - torch.exp(f)*(torch.exp(f) + z * f_x)
        
        return log_px_t, log_px_x, log_px_xx