"""
 Parameters for neural network applications
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - November 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ignite.metrics as metrics
from piqa import ssim

import dragonflai.model.neural_network_architectures.CNN as CNN
import dragonflai.model.neural_network_architectures.FCNN as FCNN
import dragonflai.model.neural_network_architectures.AE as AE
import dragonflai.model.neural_network_architectures.VAE as VAE
import dragonflai.model.neural_network_architectures.AE_clustering as AE_clust
import dragonflai.model.neural_network_architectures.VAE_clustering as VAE_clust

class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, input, target, z_mu, z_sigma):
        loss_MSE = self.mse_loss(input, target)
        loss_KLD = -0.5 * torch.sum(1 + z_sigma - z_mu.pow(2) - z_sigma.exp())

        return (loss_MSE + loss_KLD)/1000

class SSIM_loss(ssim.SSIM):   
    def forward(self,input, target):
        return 1 - super().forward(input,target)

input_size = 128*128*3
output_size = 128*128*3
#NN_model = VAE.Variational_Auto_Encoders(input_size,output_size)
#NN_model = AE_clust.Auto_Encoders_no_latent(input_size,output_size)
NN_model = VAE_clust.VAE_no_latent(input_size,output_size)

batch_size = 4
num_epoch = 100
lr = 5e-4
wd = None
optimizer = torch.optim.AdamW
#crit = SSIM_loss().cuda()
crit = VAE_loss()
#crit = nn.MSELoss()
use_scheduler = True

