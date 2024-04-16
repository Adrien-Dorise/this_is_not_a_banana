from dragonflai.model.neuralNetwork import NeuralNetwork
import torch.nn as nn
import torch.optim


    
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)



class Convolutional_Skip(NeuralNetwork):
    """Network using skip for image generation
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=512*512*3, outputs=512*512):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        dropout=0.25

        #512
        
        self.architecture.add_module('encoder_conv_start', nn.Conv2d(3,64,kernel_size=5, stride=1, padding=2))
        self.architecture.add_module("encoder_BN_start", nn.BatchNorm2d(64))
        self.architecture.add_module("encoder_relu_start", nn.ReLU())
        
        for i in range(5):
            self.architecture.add_module('encoder_conv_{}'.format(i), nn.Conv2d(64,64,kernel_size=5, stride=1, padding=2))
            self.architecture.add_module('encoder_BN_{}'.format(i), nn.BatchNorm2d(64))
            self.architecture.add_module('encoder_relu_{}'.format(i), nn.ReLU())
            
        self.architecture.add_module('encoder_conv_end', nn.Conv2d(64,3,kernel_size=5, stride=1, padding=2))
        #self.architecture.add_module('encoder_BN_end', nn.BatchNorm2d(3))
        #self.architecture.add_module('encoder_relu_end', nn.ReLU())
        
        self.architecture.apply(init_weights)
        
        
    def forward(self, data):
        out = self.architecture(data)
        out += data 
        
        return out 
    
    def set_scheduler(self,
                      scheduler=None,
                      *args, **kwargs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.75, patience=10)

    def update_scheduler(self, *args, **kwargs):
        '''update scheduler'''
        loss = kwargs['loss']
        self.scheduler.step(loss)


class CNN_Feature_Extraction(NeuralNetwork):
    """CNN architecture used for feature extraction
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.
    
    """
    def __init__(self, input_size=512*512*3, outputs=512*512):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        dropout=0.25
        channels = [16,32]

        self.architecture.add_module('conv1_1', nn.Conv2d(3,channels[0],kernel_size=3, stride=1, padding=0))
        self.architecture.add_module("conv1_BN", nn.BatchNorm2d(channels[0]))
        self.architecture.add_module("conv1_act", nn.ReLU())

        for i in range(len(channels)-1):
            self.architecture.add_module(f"conv{i}_1", nn.Conv2d(channels[i],channels[i+1],kernel_size=3, stride=1, padding=0))
            self.architecture.add_module(f"conv{i}_BN", nn.BatchNorm2d(channels[i]))
            self.architecture.add_module(f"conv{i}_act", nn.ReLU())
        

            
        self.architecture.add_module('conv_end', nn.Conv2d(channels[-1],channels[-1],kernel_size=3, stride=1, padding=0))
        self.architecture.add_module('conv_end_act', nn.Sigmoid())
        
        self.architecture.apply(init_weights)
        
        
    def forward(self, data):
        out = self.architecture(data)
        
        return out