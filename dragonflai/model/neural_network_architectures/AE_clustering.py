from dragonflai.model.neuralNetwork import NeuralNetwork
import torch.nn as nn



    
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)


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

        channels = [128,64,
                    32,16,
                    8,8]

        self.encoder.add_module('encoder_conv1', nn.Conv2d(3,channels[0],kernel_size=3, stride=1, padding=1))
        self.encoder.add_module("encoder_relu1", nn.ReLU())
        self.encoder.add_module('encoder_conv1_', nn.Conv2d(channels[0],channels[1],kernel_size=3, stride=1, padding=1))
        self.encoder.add_module("encoder_relu1_", nn.ReLU())
        self.encoder.add_module("encoder_pool1", nn.MaxPool2d(kernel_size=2, stride=2))

        self.encoder.add_module('encoder_conv2', nn.Conv2d(channels[1],channels[2],kernel_size=3, stride=1,padding=1))
        self.encoder.add_module("encoder_relu2", nn.ReLU())
        self.encoder.add_module('encoder_conv2_', nn.Conv2d(channels[2],channels[3],kernel_size=3, stride=1,padding=1))
        self.encoder.add_module("encoder_relu2_", nn.ReLU())
        self.encoder.add_module("encoder_pool2", nn.MaxPool2d(kernel_size=2, stride=2))

        self.encoder.add_module('encoder_conv3', nn.Conv2d(channels[3],channels[4],kernel_size=3, stride=1,padding=1))
        self.encoder.add_module("encoder_relu3", nn.ReLU())
        self.encoder.add_module('encoder_conv3_', nn.Conv2d(channels[4],channels[5],kernel_size=3, stride=1,padding=1))
        self.encoder.add_module("encoder_relu3_", nn.ReLU())
        self.encoder.add_module("encoder_pool3", nn.MaxPool2d(kernel_size=2, stride=2))

        #DECODER

        self.decoder.add_module("decoder_conv5", nn.ConvTranspose2d(channels[-1], channels[-3], kernel_size=2, stride=2))
        self.decoder.add_module('decoder_relu5', nn.ReLU())

        self.decoder.add_module("decoder_conv4", nn.ConvTranspose2d(channels[-3], channels[-5], kernel_size=2, stride=2))
        self.decoder.add_module('decoder_relu4', nn.ReLU())
        
        self.decoder.add_module("decoder_conv3", nn.ConvTranspose2d(channels[-5], 3, kernel_size=2, stride=2))
        self.decoder.add_module('decoder_sigmoid1', nn.Sigmoid())
        
        self.architecture.add_module("encoder",self.encoder)
        self.architecture.add_module("decoder",self.decoder)
    
    def forward(self, x):
        x = self.architecture[0](x)
        x = self.architecture[1](x)
        return x
    
    def get_encoded(self,x):
        x = self.architecture[0](x)
        return x