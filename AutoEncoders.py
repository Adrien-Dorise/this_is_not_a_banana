"""
One class classification toolbox

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""

#Tuto: https://blog.keras.io/building-autoencoders-in-keras.html
#Convolutional AE with images: https://medium.com/geekculture/face-image-reconstruction-using-autoencoders-in-keras-69a35cde01b0
#https://www.analyticsvidhya.com/blog/2021/06/complete-guide-on-how-to-use-autoencoders-in-python/

from os.path import exists
from numpy import array
from keras.models import model_from_json
from keras import Sequential
from keras import Model
from keras import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras import backend as K
import matplotlib.pyplot as plt
from cv2 import cvtColor, COLOR_BGR2RGB
import keras.utils as image



class AutoEncoders(Model):
    """
    Parameters
    ----------
    encodingDim: int
      ex: 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    
    inputPixels: int
        Number of pixels in the input pictures
    """

    def __init__(self, inputSize=[28,28], colors = 3):
         super().__init__()
         self.xSize = inputSize[0]
         self.ySize = inputSize[1]
         self.color = colors
         
        
    def build(self, optimizer='adam', loss='binary_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss)
        
    def fit(self,trainInput,validationInput, epochs = 15, batch_size = 128):
        history = self.model.fit(trainInput,trainInput,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(validationInput, validationInput))
        return history
    
    def predict(self, testInput):
        # encoded_imgs = self.encoder.predict(testInput)
        decoded_imgs = self.model.predict(testInput)
        return decoded_imgs
        
        
    def plotPrediction(self, image):
        image = array(image)
        image = image.reshape(-1,self.xSize,self.ySize,image.shape[-1])
        result = self.predict(image)
        plt.imshow(image[0,:,:,::-1], cmap=plt.get_cmap('gray')) #cmap is ignored when given an rgb image
        plt.show()
        plt.imshow(result[0,:,:,::-1], cmap=plt.get_cmap('gray')) #cmap is ignored when given an rgb image
        plt.show()
        return image
        
    def printInfos(self):
        print(self.model.summary())
        
    def saveModel(self,path):
        # serialize model to JSON
        model_json = self.model.to_json()
        iterator = 1
        while(exists(path + str(iterator) + ".json")):
            iterator+=1
        with open(path + str(iterator) + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(path + str(iterator) + ".h5")
        print("Saved model to disk")
        
        
        
    def loadModel(self, path):    
        #Load a model
        # load json and create model
        json_file = open(path+".json", 'r')
        loadedModel = json_file.read()
        json_file.close()
        loadedModel = model_from_json(loadedModel)
        # load weights into new model
        loadedModel.load_weights(path+".h5")
        print("Loaded model from disk")
        self.model = loadedModel
        
       
class ConvolutionalAutoEncoders(AutoEncoders):
    def __init__(self, inputSize=[28,28], colors = 3):
        super().__init__()
        AutoEncoders.__init__(self, inputSize, colors)
        
        self.model = Sequential()
        #Encoder
        self.model.add(Conv2D(30,3,activation='relu',padding='same', input_shape=(self.xSize,self.ySize,self.color)))
        self.model.add(MaxPooling2D(2,padding='same'))
        self.model.add(Conv2D(15,3,activation='relu',padding='same'))
        self.model.add(MaxPooling2D(2,padding='same'))
        
        #Decoder
        self.model.add(Conv2D(15,3,activation='relu',padding='same'))
        self.model.add(UpSampling2D(2))
        self.model.add(Conv2D(30,3,activation='relu',padding='same'))
        self.model.add(UpSampling2D(2))
        
        #Output
        self.model.add(Conv2D(self.color,3,activation='sigmoid',padding='same'))  


class AE(AutoEncoders):
    def __init__(self, inputSize=[28,28], colors = 3):
        super().__init__()
        AutoEncoders.__init__(self, inputSize, colors)
        
        self.model = Sequential()
        #Encoder
        self.model.add(Input(shape = (inputSize[0]* inputSize[1]*colors)))
        # self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        
        #Decoder
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        # self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(inputSize[0]* inputSize[1]*colors, activation ='sigmoid'))
        


# a = AutoEncoders([28,28])
# a.build()

# from keras.datasets import mnist
# import numpy as np
# (x_train, _), (x_test, _) = mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# print(x_train.shape)
# print(x_test.shape)


# a.fit(x_train,x_test,epochs=15)
# a.printInfos()

# # Encode and decode some digits
# # Note that we take them from the *test* set
# img = a.predict(x_test)
# a.plotPrediction(x_test, img)















