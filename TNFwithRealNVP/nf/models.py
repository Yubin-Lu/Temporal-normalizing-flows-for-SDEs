import torch
import torch.nn as nn
import numpy as np


class NormalizingFlowModel(nn.Module):

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        for flow in self.flows:
            x, ld, px = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det, px

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples, t):
        z = self.prior.sample((n_samples,))  
        time = np.ones([n_samples,1])*t
        zz = torch.Tensor(np.concatenate((z,time),axis=1))
        x, _ = self.inverse(zz)
        return x

    def sample_px(self, x, t):
        #Using tNFs to estimate time-varying density p(x,t) 
        n_samples, _ = x.shape
        time = np.ones([n_samples,1])*t
        xx = torch.Tensor(np.concatenate((x,time),axis=1))
        # print(xx.shape)
        z, prior_logprob, log_det, px = self.forward(xx)
        return px
