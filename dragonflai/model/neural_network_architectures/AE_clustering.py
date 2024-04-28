from dragonflai.model.neuralNetwork import NeuralNetwork
import torch.nn as nn



    
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
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)
    
class UpTranspose(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

    def forward(self, x):
        return self.layer(x)


class Auto_Encoders_latent(NeuralNetwork):
    """Auto_encoders for image generation
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=512*512*3, outputs=512*512):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        self.encoder = nn.Sequential().to(self.device)
        self.decoder = nn.Sequential().to(self.device)
        self.latent = nn.Sequential().to(self.device)

        channels = [128,64,32,16]

        #ENCODER
        self.encoder.add_module('down1', Down(3,channels[0]))
        self.encoder.add_module('down2', Down(channels[0],channels[1]))
        self.encoder.add_module('down3', Down(channels[1],channels[2]))        
        self.encoder.add_module('down4', Down(channels[2],channels[3]))        


        #LATENT
        latent_dim = 1024
        latent_img_shape = 8
        latent_channels= channels[-1]

        self.latent.add_module("flatten1",nn.Flatten())
        
        self.latent.add_module("latent1", nn.Linear(latent_channels*latent_img_shape*latent_img_shape, latent_dim))
        self.latent.add_module("latent_relu1", nn.ReLU())

        
        self.latent.add_module("latent2", nn.Linear(latent_dim,latent_channels*latent_img_shape*latent_img_shape))
        self.latent.add_module("latent_relu2", nn.ReLU())
        
        self.latent.add_module("latent_reshape2", nn.Unflatten(1,(latent_channels,latent_img_shape,latent_img_shape)))
        

        #DECODER

        self.decoder.add_module("up1", Up(channels[-1],channels[-1]))
        self.decoder.add_module("up2", Up(channels[-1],channels[-2]))
        self.decoder.add_module("up3", Up(channels[-2],channels[-3]))
        self.decoder.add_module("up4", Up(channels[-3],channels[-4]))
        self.decoder.add_module("up5_1", nn.Conv2d(channels[-4], 3, kernel_size=5, stride=1, padding=2, bias=False))
        self.decoder.add_module('up5_2', nn.Sigmoid())
        
        self.architecture.add_module("encoder",self.encoder)
        self.architecture.add_module("latent",self.latent)
        self.architecture.add_module("decoder",self.decoder)
    
    def forward(self, x):
        x = self.architecture[0](x)
        x = self.architecture[1](x)
        x = self.architecture[2](x)
        return x
    
    def get_encoded(self,x):
        x = self.architecture[0](x)
        return x

class Auto_Encoders_no_latent(NeuralNetwork):
    """Auto_encoders for image generation
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=512*512*3, outputs=512*512):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        self.encoder = nn.Sequential().to(self.device)
        self.decoder = nn.Sequential().to(self.device)    

        channels = [256,256,128,128]

        #ENCODER
        self.encoder.add_module('down1', Down(3,channels[0]))
        self.encoder.add_module('down2', Down(channels[0],channels[1]))
        self.encoder.add_module('down3', Down(channels[1],channels[2]))        
        self.encoder.add_module('down4', Down(channels[2],channels[3]))        

        #DECODER

        self.decoder.add_module("up1", Up(channels[-1],channels[-1]))
        self.decoder.add_module("up2", Up(channels[-1],channels[-2]))
        self.decoder.add_module("up3", Up(channels[-2],channels[-3]))
        self.decoder.add_module("up4", Up(channels[-3],channels[-4]))
        self.decoder.add_module("up5_1", nn.Conv2d(channels[-4], 3, kernel_size=5, stride=1, padding=2, bias=False))
        self.decoder.add_module('up5_2', nn.Sigmoid())
        
        self.architecture.add_module("encoder",self.encoder)
        self.architecture.add_module("decoder",self.decoder)
    
    def forward(self, x):
        x = self.architecture[0](x)
        x = self.architecture[1](x)
        return x
    
    def get_encoded(self,x):
        x = self.architecture[0](x)
        return x
    

class Auto_Encoders_no_latent_transpose(NeuralNetwork):
    """Auto_encoders for image generation
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=512*512*3, outputs=512*512):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        self.encoder = nn.Sequential().to(self.device)
        self.decoder = nn.Sequential().to(self.device)    

        channels = [128,64,32,16]

        #ENCODER
        self.encoder.add_module('down1', Down(3,channels[0]))
        self.encoder.add_module('down2', Down(channels[0],channels[1]))
        self.encoder.add_module('down3', Down(channels[1],channels[2]))        
        self.encoder.add_module('down4', Down(channels[2],channels[3]))        

        #DECODER
        self.decoder.add_module("up1", Up(channels[-1],channels[-2]))
        self.decoder.add_module("up2", Up(channels[-2],channels[-3]))
        self.decoder.add_module("up3", Up(channels[-3],channels[-4]))
        self.decoder.add_module("up4_1", nn.ConvTranspose2d(channels[-4], 3, kernel_size=2, stride=2))
        self.decoder.add_module('up4_2', nn.ReLU())
        
        self.architecture.add_module("encoder",self.encoder)
        self.architecture.add_module("decoder",self.decoder)
    
    def forward(self, x):
        x = self.architecture[0](x)
        x = self.architecture[1](x)
        return x
    
    def get_encoded(self,x):
        x = self.architecture[0](x)
        return x