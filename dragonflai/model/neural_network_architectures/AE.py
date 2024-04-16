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

        dropout=0.0
        channels = [256,128,32,8]
        
        self.architecture.add_module('encoder_conv1', nn.Conv2d(3,channels[0],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu1", nn.ReLU())
        self.architecture.add_module('encoder_conv1_', nn.Conv2d(channels[0],channels[1],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu1_", nn.ReLU())
        self.architecture.add_module("encoder_pool1", nn.MaxPool2d(kernel_size=2, stride=2))
        #self.architecture.add_module('encoder_dropout1', nn.Dropout2d(p=dropout))

        self.architecture.add_module('encoder_conv2', nn.Conv2d(channels[1],channels[2],kernel_size=3, stride=1,padding=1))
        self.architecture.add_module("encoder_relu2", nn.ReLU())
        self.architecture.add_module('encoder_conv2_', nn.Conv2d(channels[2],channels[3],kernel_size=3, stride=1,padding=1))
        self.architecture.add_module("encoder_relu2_", nn.ReLU())
        self.architecture.add_module("encoder_pool2", nn.MaxPool2d(kernel_size=2, stride=2))
        #self.architecture.add_module('encoder_dropout2', nn.Dropout2d(p=dropout))

        #DECODER

        self.architecture.add_module("decoder_conv5", nn.ConvTranspose2d(channels[-1], channels[-3], kernel_size=2, stride=2))
        self.architecture.add_module('decoder_relu5', nn.ReLU())
        #self.architecture.add_module('decoder_dropout5', nn.Dropout2d(p=dropout))
        
        self.architecture.add_module("decoder_conv4", nn.ConvTranspose2d(channels[-3], 3, kernel_size=2, stride=2))
        self.architecture.add_module('decoder_sigmoid1', nn.Sigmoid())
        




class Auto_Encoders_latent1(NeuralNetwork):
    """Auto_encoders for image generation
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=512*512*3, outputs=512*512):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        dropout = 0.1
        channels = [32,64,
                    128,256,
                    256,512,
                    512,512]

        #512

        self.architecture.add_module('encoder_conv1_1', nn.Conv2d(3,channels[0],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu1_1", nn.ReLU())
        self.architecture.add_module('encoder_conv1_2', nn.Conv2d(channels[0],channels[1],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu1_2", nn.ReLU())
        self.architecture.add_module("encoder_pool1", nn.MaxPool2d(kernel_size=2, stride=2))


        #256


        self.architecture.add_module('encoder_conv2_1', nn.Conv2d(channels[1],channels[2],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu2_1", nn.ReLU())
        self.architecture.add_module('encoder_conv2_2', nn.Conv2d(channels[2],channels[3],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu2_2", nn.ReLU())
        self.architecture.add_module("encoder_pool2", nn.MaxPool2d(kernel_size=2, stride=2))

        #128

        self.architecture.add_module('encoder_conv3_1', nn.Conv2d(channels[3],channels[4],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu3_1", nn.ReLU())
        self.architecture.add_module('encoder_conv3_2', nn.Conv2d(channels[4],channels[5],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu3_2", nn.ReLU())
        self.architecture.add_module("encoder_pool3", nn.MaxPool2d(kernel_size=2, stride=2))
        
        #64

        self.architecture.add_module('encoder_conv4_1', nn.Conv2d(channels[5],channels[6],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu4_1", nn.ReLU())
        self.architecture.add_module('encoder_conv4_2', nn.Conv2d(channels[6],channels[7],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu4_2", nn.ReLU())
        self.architecture.add_module("encoder_pool4", nn.MaxPool2d(kernel_size=2, stride=2))

        #32


        #LATENT
        latent_dim = 2054
        latent_img_shape = 8
        latent_channels= channels[-1]

        self.architecture.add_module("flatten1",nn.Flatten())
        
        self.architecture.add_module("latent1", nn.Linear(latent_channels*latent_img_shape*latent_img_shape, latent_dim))
        self.architecture.add_module("latent_relu1", nn.ReLU())
        #self.architecture.add_module('latent_dropout1', nn.Dropout(p=dropout))

        
        self.architecture.add_module("latent2", nn.Linear(latent_dim,latent_channels*latent_img_shape*latent_img_shape))
        self.architecture.add_module("latent_relu2", nn.ReLU())
        #self.architecture.add_module('latent_dropout2', nn.Dropout(p=dropout))
        
        self.architecture.add_module("latent_reshape2", nn.Unflatten(1,(latent_channels,latent_img_shape,latent_img_shape)))
        
        #DECODER
                
        #32
        
        self.architecture.add_module("decoder_conv4_1", nn.ConvTranspose2d(channels[-1], channels[-3], kernel_size=5, stride=2, padding=2, output_padding=1))
        self.architecture.add_module('decoder_relu4_1', nn.ReLU())

        #64
        
        self.architecture.add_module("decoder_conv3_1", nn.ConvTranspose2d(channels[-3], channels[-5], kernel_size=5, stride=2, padding=2, output_padding=1))
        self.architecture.add_module('decoder_relu3_1', nn.ReLU())
        
        #128

        self.architecture.add_module("decoder_conv2_1", nn.ConvTranspose2d(channels[-5], channels[-7], kernel_size=5, stride=2, padding=2, output_padding=1))
        self.architecture.add_module('decoder_relu2_1', nn.ReLU())

        
        #254

        self.architecture.add_module("decoder_conv1_1", nn.ConvTranspose2d(channels[-7], 3, kernel_size=5, stride=2, padding=2, output_padding=1))
        self.architecture.add_module('decoder_sigmoid1', nn.Sigmoid())
        
        #512

        self.architecture.apply(init_weights)
        

        
class Auto_Encoders_latent2(NeuralNetwork):
    """Auto_encoders for image generation
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=512*512*3, outputs=512*512):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        dropout = 0.1
        channels = [32,64,64,64]


        #512

        self.architecture.add_module('encoder_conv1_1', nn.Conv2d(3,channels[0],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu1_1", nn.ReLU())
        self.architecture.add_module("encoder_pool1", nn.MaxPool2d(kernel_size=2, stride=2))


        #256


        self.architecture.add_module('encoder_conv2_1', nn.Conv2d(channels[0],channels[1],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu2_1", nn.ReLU())
        self.architecture.add_module("encoder_pool2", nn.MaxPool2d(kernel_size=2, stride=2))

        #128

        self.architecture.add_module('encoder_conv3_1', nn.Conv2d(channels[1],channels[2],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu3_1", nn.ReLU())
        self.architecture.add_module("encoder_pool3", nn.MaxPool2d(kernel_size=2, stride=2))
        
        #64

        self.architecture.add_module('encoder_conv4_1', nn.Conv2d(channels[2],channels[3],kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_relu4_1", nn.ReLU())
        self.architecture.add_module("encoder_pool4", nn.MaxPool2d(kernel_size=2, stride=2))

        #32

        
        #LATENT
        latent_dim = 512
        latent_img_shape = 8
        latent_channels = channels[-1]

        self.architecture.add_module("flatten1",nn.Flatten())
        
        self.architecture.add_module("latent1", nn.Linear(latent_channels*latent_img_shape*latent_img_shape, latent_dim))
        self.architecture.add_module("latent_relu1", nn.ReLU())
        #self.architecture.add_module('latent_dropout1', nn.Dropout(p=dropout))

        
        self.architecture.add_module("latent2", nn.Linear(latent_dim,latent_channels*latent_img_shape*latent_img_shape))
        self.architecture.add_module("latent_relu2", nn.ReLU())
        #self.architecture.add_module('latent_dropout2', nn.Dropout(p=dropout))
        
        self.architecture.add_module("latent_reshape2", nn.Unflatten(1,(latent_channels,latent_img_shape,latent_img_shape)))
        
        #DECODER
                
        #32
        
        self.architecture.add_module("decoder_conv4_1", nn.ConvTranspose2d(channels[-1], channels[-2], kernel_size=5, stride=2, padding=2, output_padding=1))
        self.architecture.add_module('decoder_relu4_1', nn.ReLU())

        #64
        
        self.architecture.add_module("decoder_conv3_1", nn.ConvTranspose2d(channels[-2], channels[-3], kernel_size=5, stride=2, padding=2, output_padding=1))
        self.architecture.add_module('decoder_relu3_1', nn.ReLU())
        
        #128

        self.architecture.add_module("decoder_conv2_1", nn.ConvTranspose2d(channels[-3], channels[-4], kernel_size=5, stride=2, padding=2, output_padding=1))
        self.architecture.add_module('decoder_relu2_1', nn.ReLU())

        
        #254

        self.architecture.add_module("decoder_conv1_1", nn.ConvTranspose2d(channels[-4], 3, kernel_size=5, stride=2, padding=2, output_padding=1))
        self.architecture.add_module('decoder_final_act', nn.Sigmoid())
        
        #512

        self.architecture.apply(init_weights)

