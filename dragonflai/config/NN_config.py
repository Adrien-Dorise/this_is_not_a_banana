"""
 Parameters for neural network applications
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - November 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dragonflai.model.neural_network_architectures.CNN as CNN
import dragonflai.model.neural_network_architectures.FCNN as FCNN
import dragonflai.model.neural_network_architectures.AE as AE
import dragonflai.model.neural_network_architectures.VAE as VAE


input_size = 128*128*3
output_size = 128*128*3
#NN_model = VAE.Variational_Auto_Encoders(input_size,output_size)
NN_model = AE.Auto_Encoders_no_latent(input_size,output_size)

batch_size = 4
num_epoch = 4
lr = 1e-3
wd = 1e-3
optimizer = torch.optim.AdamW
crit = nn.L1Loss()
# crit = VAE.VAE_loss()
