from dragonflai.model.neuralNetwork import NeuralNetwork
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np


    
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)

class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, input, target, z_mu, z_sigma):
        loss_MSE = self.mse_loss(input, target)
        loss_KLD = -0.5 * torch.sum(1 + z_sigma - z_mu.pow(2) - z_sigma.exp())

        return (loss_MSE + loss_KLD)/1000

class Variational_Auto_Encoders(NeuralNetwork):
    """Auto_encoders for image generation
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=512*512*3, outputs=512*512):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        dropout = 0.4
        channels = [64,128,256,256]
        self.encoder = nn.Sequential().to(self.device)
        self.decoder = nn.Sequential().to(self.device)        

        #512
        
        self.encoder.add_module('encoder_conv1_1', nn.Conv2d(3,channels[0],kernel_size=3, stride=1, padding=1))
        self.encoder.add_module("encoder_relu1_1", nn.ReLU())
        self.encoder.add_module("encoder_BN1", nn.BatchNorm2d(channels[0]))
        self.encoder.add_module("encoder_pool1", nn.MaxPool2d(kernel_size=2, stride=2))

        #256

        self.encoder.add_module('encoder_conv2_1', nn.Conv2d(channels[0],channels[1],kernel_size=3, stride=1, padding=1))
        self.encoder.add_module("encoder_relu2_1", nn.ReLU())
        self.encoder.add_module("encoder_BN2", nn.BatchNorm2d(channels[1]))
        self.encoder.add_module("encoder_pool2", nn.MaxPool2d(kernel_size=2, stride=2))

        #128

        self.encoder.add_module('encoder_conv3_1', nn.Conv2d(channels[1],channels[2],kernel_size=3, stride=1, padding=1))
        self.encoder.add_module("encoder_relu3_1", nn.ReLU())
        self.encoder.add_module("encoder_BN3", nn.BatchNorm2d(channels[2]))
        self.encoder.add_module("encoder_pool3", nn.MaxPool2d(kernel_size=2, stride=2))
        
        #64

        self.encoder.add_module('encoder_conv4_1', nn.Conv2d(channels[2],channels[3],kernel_size=3, stride=1, padding=1))
        self.encoder.add_module("encoder_relu4_1", nn.ReLU())
        self.encoder.add_module("encoder_BN4", nn.BatchNorm2d(channels[3]))
        self.encoder.add_module("encoder_pool4", nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder.add_module("encoder_flatten1", nn.Flatten())

        self.encoder.apply(init_weights)
        
        #32
        
        #LATENT
        
        latent_img_shape = 8
        latent_channels = channels[-1]

        self.z_mean = nn.Linear(latent_channels*latent_img_shape*latent_img_shape, 2).to(self.device)
        self.z_log_var = nn.Linear(latent_channels*latent_img_shape*latent_img_shape, 2).to(self.device)
        
        #DECODER        
        
        self.decoder.add_module("latent1", nn.Linear(2, latent_channels*latent_img_shape*latent_img_shape))
        self.decoder.add_module("latent_relu1", nn.ReLU())
        #self.decoder.add_module("dropout1", nn.Dropout(dropout))
        self.decoder.add_module("latent_reshape2", nn.Unflatten(1,(latent_channels,latent_img_shape,latent_img_shape)))                
        
        #32
        
        self.decoder.add_module("decoder_conv4_1", nn.ConvTranspose2d(channels[-1], channels[-2], kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder.add_module('decoder_relu4_1', nn.ReLU())

        #64
        
        self.decoder.add_module("decoder_conv3_1", nn.ConvTranspose2d(channels[-2], channels[-3], kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder.add_module('decoder_relu3_1', nn.ReLU())
        
        #128

        self.decoder.add_module("decoder_conv2_1", nn.ConvTranspose2d(channels[-3], channels[-4], kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder.add_module('decoder_relu2_1', nn.ReLU())

        #254

        self.decoder.add_module("decoder_conv1_1", nn.ConvTranspose2d(channels[-4], 3, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder.add_module('decoder_sigmoid1', nn.Sigmoid())
        
        #512

        self.decoder.apply(init_weights)
         
    
    def reparameterize(self, z_mean, z_log_var):
        std = z_log_var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(z_mean)
    
    def encoding(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.sampling(z_mean, z_log_var)
        return encoded
    
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded
      
            
    def loss_calculation(self, crit, inputs, target, *args, **kwargs):
        '''compute loss'''
        
        encoded, z_mu, z_sigma, outputs = self.forward(inputs)
        loss    = crit(outputs, target, z_mu, z_sigma)

        return loss, outputs
    
    def set_optimizer(self, 
                      optimizer=torch.optim.AdamW, 
                      learning_rate=1e-4, 
                      weight_decay=1e-4, 
                      *args, **kwargs):
        '''set optimizer'''
        self.opt = [None, None]
        if weight_decay is None:
            self.opt[0] = optimizer(self.encoder.parameters(), lr=learning_rate)
            self.opt[1] = optimizer(self.decoder.parameters(), lr=learning_rate)
        else:
            self.opt[0] = optimizer(self.encoder.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay)
            self.opt[1] = optimizer(self.decoder.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay)
        
        self.scaler = []
        self.scaler.append(torch.cuda.amp.GradScaler(enabled=self.use_gpu))
        
    def set_scheduler(self, scheduler=None, *args, **kwargs):
        '''set scheduler'''
        self.scheduler = []
        self.scheduler.append(torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt[0], mode='min', factor=0.95, patience=30))
        self.scheduler.append(torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt[1], mode='min', factor=0.95, patience=30))
        #self.scheduler = [None]
        


      
        

    
    