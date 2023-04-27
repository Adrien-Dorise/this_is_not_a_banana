"""
One class classification toolbox

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""

#Tuto: https://blog.keras.io/building-autoencoders-in-keras.html
#Convolutional AE with images: https://medium.com/geekculture/face-image-reconstruction-using-autoencoders-in-keras-69a35cde01b0
#https://www.analyticsvidhya.com/blog/2021/06/complete-guide-on-how-to-use-autoencoders-in-python/

from os.path import exists
from numpy import array, prod
from keras.models import model_from_json
from tensorflow import squeeze
from keras import Sequential, Model, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, UpSampling2D, InputLayer
from keras import backend as K
import matplotlib.pyplot as plt
from cv2 import cvtColor, COLOR_BGR2RGB
import keras.utils as image
from math import sqrt
import numpy as np
import pandas as pd



class NeuralNetwork(Model):
    """
    Parameters
    ----------
    encodingDim: int
      ex: 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    
    inputPixels: int
        Number of pixels in the input pictures
    """

    def __init__(self, inputShape = (28,28,3)):
         super().__init__()
         self.inputShape = inputShape

        
    def build(self, optimizer='adam', loss='binary_crossentropy', lr = 0.001):
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.optimizer.learning_rate = lr
        
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
        image = image.reshape(-1,self.inputShape[0],self.inputShape[1], self.inputShape[2])
        result = self.predict(image)
        plt.imshow(image[0,:,:,::-1], cmap=plt.get_cmap('gray')) #cmap is ignored when given an rgb image
        plt.show()
        plt.imshow(result[0,:,:,::-1], cmap=plt.get_cmap('gray')) #cmap is ignored when given an rgb image
        plt.show()
        return result
    
    
    
    def printInfos(self):
        print(f"Input shape: {self.inputShape}")
        print(self.model.summary())
        
    def saveModel(self,path, history, overwrite=False):
        # serialize model to JSON
        model_json = self.model.to_json()
        iterator = 1
        if not overwrite:
            while(exists(path + str(iterator) + ".json")):
                iterator+=1
        with open(path + str(iterator) + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(path + str(iterator) + ".h5")
        history.to_csv(path + str(iterator) + ".csv")
        print("Saved model to " + path + str(iterator))
        
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
        
        history = pd.read_csv(path + ".csv")
        return history
        

        

class ConvolutionalAutoEncoders(NeuralNetwork):
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        NeuralNetwork.__init__(self, inputShape)
        
        self.modelName = "CNN auto-encoders"
        
        self.model = Sequential()
        #Encoder
        self.model.add(Conv2D(30,3,activation='relu',padding='same', input_shape=inputShape))
        self.model.add(MaxPooling2D(2,padding='same'))
        self.model.add(Conv2D(15,3,activation='relu',padding='same'))
        self.model.add(MaxPooling2D(2,padding='same'))
        
        #Decoder
        self.model.add(Conv2D(15,3,activation='relu',padding='same'))
        self.model.add(UpSampling2D(2))
        self.model.add(Conv2D(30,3,activation='relu',padding='same'))
        self.model.add(UpSampling2D(2))
        
        #Output
        self.model.add(Conv2D(self.inputShape[-1],3,activation='sigmoid',padding='same'))  


class AE(NeuralNetwork):
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        NeuralNetwork.__init__(self, inputShape)
        
        self.modelName = "fully connected auto-encoders"
        
        dropout=0.2
        self.model = Sequential()
        
        #Encoder
        # self.model.add(InputLayer(inputShape))
        # self.model.add(Flatten())
        self.model.add(Input(shape=((int)(prod(inputShape)))))
        
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(dropout))
        
        #Decoder
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(dropout))
        
        # self.model.add(Dense(prod(inputShape), activation='relu'))
        # self.model.add(Reshape(inputShape))        
        self.model.add(Dense(prod(inputShape), activation=None))
        
class AECNN(NeuralNetwork):
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        NeuralNetwork.__init__(self, inputShape)
        
        self.modelName = "fully connected + CNN auto-encoders"
        
        dropout=0.25
        self.model = Sequential()
        
        #CNN encoder
        self.model.add(Conv2D(8,5,activation='relu',padding='same', input_shape=inputShape))
        self.model.add(MaxPooling2D(2,padding='same'))
        self.model.add(Dropout(dropout))
        self.model.add(Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(MaxPooling2D(2,padding='same'))
        self.model.add(Dropout(dropout))
        self.model.add(Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(MaxPooling2D(2,padding='same'))
        self.model.add(Dropout(dropout))
        self.model.add(Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(MaxPooling2D(2,padding='same'))
        self.model.add(Dropout(dropout))

        
        #Flatten
        shape = self.model.output_shape
        shapeFlat = np.prod(shape[1:])
        self.model.add(Flatten())
        
        
        #Encoder
        self.model.add(Dense(shapeFlat, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(dropout))
        
        self.model.add(Dense(258, activation='relu'))
        self.model.add(Dropout(dropout))
        
        #Decoder
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(shapeFlat, activation='relu'))
        self.model.add(Dropout(dropout))
        
        #CNN decoder
        self.model.add(Reshape(shape[1:])) 
        
        self.model.add(Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(UpSampling2D(2))
        self.model.add(Dropout(dropout))
        self.model.add(Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(UpSampling2D(2))
        self.model.add(Dropout(dropout))
        self.model.add(Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(UpSampling2D(2))
        self.model.add(Dropout(dropout))
        self.model.add(Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(UpSampling2D(2))

        
        #Output
        self.model.add(Conv2D(self.inputShape[-1],3,activation=None,padding='same'))  
        
    
def score(target,prediction):
    score = 0
    for i in range(len(target)):
        diff = target[i] - prediction[i]
        score += sqrt(diff ** 2)
    return score


if __name__ == "__main__":
    a = NeuralNetwork([28,28])
    a.build()
    
    from keras.datasets import mnist
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    print(x_train.shape)
    print(x_test.shape)
    
    
    a.fit(x_train,x_test,epochs=15)
    a.printInfos()
    
    # Encode and decode some digits
    # Note that we take them from the *test* set
    img = a.predict(x_test)
    a.plotPrediction(x_test, img)
    
    













