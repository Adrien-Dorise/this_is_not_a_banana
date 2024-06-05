from dragonflai.model.neuralNetwork import NeuralNetwork
import torch.nn as nn
from torch.autograd import Variable
import torch


    
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)


class DoubleDown(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)
    

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer.apply(init_weights)

    def forward(self, x):
        return self.layer(x)
    
class UpTranspose(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),
            nn.ReLU()
        )
        self.layer.apply(init_weights)

    def forward(self, x):
        return self.layer(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.layer.apply(init_weights)

    def forward(self, x):
        return self.layer(x)




class VAE_no_latent(NeuralNetwork):
    """VAE for image generation
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=512*512*3, outputs=512*512):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        self.encoder = nn.Sequential().to(self.device)
        self.decoder = nn.Sequential().to(self.device)    

        channels = [64,64,128,128]

        #ENCODER
        self.encoder.add_module('down1', Down(3,channels[0]))
        self.encoder.add_module('down2', Down(channels[0],channels[1]))
        self.encoder.add_module('down3', Down(channels[1],channels[2]))        
        self.encoder.add_module('down4', Down(channels[2],channels[3]))


        #LATENT
        latent_img_shape = 8
        latent_channels= channels[-1]
        
        self.flatten = nn.Flatten()
        self.z_mean = nn.Linear(latent_channels*latent_img_shape*latent_img_shape, 2).to(self.device)
        self.z_log_var = nn.Linear(latent_channels*latent_img_shape*latent_img_shape, 2).to(self.device)

        #DECODER

        self.decoder.add_module("latent1", nn.Linear(2, latent_channels*latent_img_shape*latent_img_shape))
        self.decoder.add_module("latent_relu1", nn.ReLU())
        #self.decoder.add_module("dropout1", nn.Dropout(dropout))
        self.decoder.add_module("latent_reshape2", nn.Unflatten(1,(latent_channels,latent_img_shape,latent_img_shape)))           
        
        self.decoder.add_module("up1", Up(channels[-1],channels[-1]))
        self.decoder.add_module("up2", Up(channels[-1],channels[-2]))
        self.decoder.add_module("up3", Up(channels[-2],channels[-3]))
        self.decoder.add_module("up4", Up(channels[-3],channels[-4]))
        self.decoder.add_module("up5_1", nn.Conv2d(channels[-3], 3, kernel_size=5, stride=1, padding=2, bias=False))
        self.decoder.add_module('up5_2', nn.Sigmoid())
        
        self.architecture.add_module("encoder",self.encoder)
        self.architecture.add_module("decoder",self.decoder)
    
    def reparameterize(self, z_mean, z_log_var):
        std = z_log_var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(z_mean)
    
    def encoding(self, x):
        x = self.architecture[0](x)
        x = nn.Flatten(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.sampling(z_mean, z_log_var)
        return encoded
    
    def forward(self, x):
        x = self.architecture[0](x)
        x = self.flatten(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.architecture[1](encoded)
        return encoded, z_mean, z_log_var, decoded

    def get_encoded(self,x):
        x = self.architecture[0](x)
        return x
    
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
    
