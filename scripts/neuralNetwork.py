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
import tensorflow as tf
from keras import Sequential, Model, Input
from keras import layers as layers 
from keras import backend as K
from keras import metrics as metrics
import keras.losses as losses
from keras.engine import data_adapter
import matplotlib.pyplot as plt
from cv2 import cvtColor, COLOR_BGR2RGB
import keras.utils as image
from math import sqrt
import numpy as np
import pandas as pd
import pickle



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
        
    def save_weights(self, path, history, overwrite=False):
        iterator = 1
        if not overwrite:
            while(exists(path + str(iterator) + ".json")):
                iterator+=1
        self.model.save_weights(path + str(iterator) + ".h5")
        history.to_csv(path + str(iterator) + ".csv")
        print("Saved model to " + path + str(iterator))
        
    def load_weights(self, path):
        self.model.load_weights(path+".h5")
        history = pd.read_csv(path + ".csv")
        print("Loaded model from disk")
        
        return history
        
    def saveModel_json(self, path, history, overwrite=False):
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
        
    def loadModel_json(self, path):    
        #Load a model
        # load json and create model
        json_file = open(path+".json", 'r')
        loadedModel = json_file.read()
        json_file.close()
        loadedModel = model_from_json(loadedModel)
        # load weights into new model
        loadedModel.load_weights(path+".h5")
        history = pd.read_csv(path + ".csv")
        self.model = loadedModel
        print("Loaded model from disk")
        
        return history
        
    def saveModel_pkl(self, path, history, overwrite=False):
        iterator = 1
        if not overwrite:
            while(exists(path + str(iterator) + ".json")):
                iterator+=1
        with open(path + str(iterator) + ".pkl", "wb") as file:
            pickle.dump(self.model,file)
        # serialize weights to HDF5
        self.model.save_weights(path + str(iterator) + ".h5")
        history.to_csv(path + str(iterator) + ".csv")
        print("Saved model to " + path + str(iterator))

        
    def loadModel_pkl(self, path):    
        #Load a model
        with open(path + ".pkl", "rb") as file:
            self.model = pickle.load(file)
        # load weights into new model
        self.model.load_weights(path+".h5")
        history = pd.read_csv(path + ".csv")
        print("Loaded model from disk")
        
        return history
    
    
class VAE_MODEL(NeuralNetwork):
   
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        NeuralNetwork.__init__(self, inputShape)
        self.inputShape = inputShape
        self.loss = losses.binary_crossentropy
        
    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
           z_mean, z_log_var = inputs
           batch = K.shape(z_mean)[0]
           dim = K.shape(z_mean)[1]
           epsilon = tf.random.normal(shape=(batch, dim))
           return z_mean + K.exp(0.5 * z_log_var) * epsilon
       
    def loss_vae(self, inputs, outputs, z_mean, z_log_var, input_shape, loss):
        reconstruction_loss = loss(K.flatten(inputs), K.flatten(outputs))
        reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2]
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        B = 1000   
        vae_loss = K.mean(B * reconstruction_loss + kl_loss)
        return [vae_loss, reconstruction_loss, kl_loss]

    def fit(self,trainInput,validationInput, epochs = 15, batch_size = 128):
        history = self.model.fit(trainInput,None,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validationInput, None))
        return history


    def build(self, optimizer='adam', loss='binary_crossentropy', lr = 0.001):
        if loss == 'binary_crossentropy':
            self.loss = losses.binary_crossentropy
        elif loss == 'mae':
            self.loss = losses.mae
        elif loss == 'mse':
            self.loss = losses.mse
        self.model.compile(optimizer=optimizer)
        self.model.optimizer.learning_rate = lr
        
        
        
class VAE_LAYER(Model):
   
    def __init__(self, encoder, latent, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        
    
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        latent_output = self.latent(z)
        reconstruction = self.decoder(latent_output)
        return reconstruction
    
    @property
    #https://keras.io/examples/generative/vae/
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]    
    
    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.latent(z)
            reconstruction = self.decoder(reconstruction)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



class ConvolutionalAutoEncoders(NeuralNetwork):
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        NeuralNetwork.__init__(self, inputShape)
        
        self.modelName = "ConvolutionalAutoEncoders"
        dropout=0.2
        
        self.model = Sequential()
        #Encoder
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same', input_shape=inputShape))
        self.model.add(layers.AveragePooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.AveragePooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.AveragePooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.AveragePooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.AveragePooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))

        
        #Decoder
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.UpSampling2D(2))
        
        #Output
        self.model.add(layers.Conv2D(self.inputShape[-1],3,activation='sigmoid',padding='same'))  


class ConvolutionalAutoEncoders2(NeuralNetwork):
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        NeuralNetwork.__init__(self, inputShape)
        
        self.modelName = "ConvolutionalAutoEncoders2"
        dropout=0.2
        
        self.model = Sequential()
        
        self.model.add(layers.Conv2D(64, (2, 2), strides = 1, padding = 'same', input_shape = inputShape))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(32, (2, 2), strides = 1, padding = 'same'))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(16, (2, 2), strides = 1, padding = 'same'))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        
        #latent
        #self.model.add(Conv2D(8, (2, 2), strides = 1, padding = 'same'))
        #self.model.add(Dropout(dropout))
        
        #Flatten
        shape = self.model.output_shape[1:]
        shapeFlat = np.prod(shape)
        self.model.add(layers.Flatten())
        
        self.model.add(layers.Dense(shapeFlat, activation='relu'))

        #CNN decoder
        self.model.add(layers.Reshape(shape)) 
        
        #decode
        self.model.add(layers.Conv2DTranspose(16, (2, 2), strides = 1, padding = 'same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2DTranspose(32, (2, 2), strides = 1, padding = 'same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2DTranspose(64, (2, 2), strides = 1, padding = 'same'))
        self.model.add(layers.UpSampling2D(2))
    
        self.model.add(layers.Conv2DTranspose(self.inputShape[-1], (1, 1), strides = 1, activation = 'sigmoid', padding = 'same'))
        
class ConvolutionalAutoEncoders3(NeuralNetwork):
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        NeuralNetwork.__init__(self, inputShape)
        
        self.modelName = "ConvolutionalAutoEncoders3"
        dropout=0.2
        
        self.model = Sequential()
        
        self.model.add(layers.Conv2D(32, (3, 3), strides = 1, padding = 'same', input_shape = inputShape))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(64, (3, 3), strides = 1, padding = 'same'))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(64, (3, 3), strides = 1, padding = 'same'))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(64, (3, 3), strides = 1, padding = 'same'))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Dropout(dropout))
        
        #latent
        #self.model.add(layers.Conv2D(8, (3, 3), strides = 1, padding = 'same'))
        #self.model.add(layers.Dropout(dropout))
        
        #Flatten
        shape = self.model.output_shape[1:]
        shapeFlat = np.prod(shape)
        self.model.add(layers.Flatten())
        
        self.model.add(layers.Dense(shapeFlat, activation='relu'))

        #CNN decoder
        self.model.add(layers.Reshape(shape)) 
        
        #decode
        self.model.add(layers.Conv2DTranspose(64, (3, 3), strides = 1, padding = 'same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2DTranspose(64, (3, 3), strides = 1, padding = 'same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2DTranspose(64, (3, 3), strides = 1, padding = 'same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2DTranspose(32, (3, 3), strides = 1, padding = 'same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
    
        self.model.add(layers.Conv2DTranspose(self.inputShape[-1], (1, 1), strides = 1, activation='sigmoid', padding = 'same'))

class AE(NeuralNetwork):
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        NeuralNetwork.__init__(self, inputShape)
        
        self.modelName = "AE"
        
        dropout=0.2
        self.model = Sequential()
        
        #Encoder
        # self.model.add(layers.InputLayer(inputShape))
        # self.model.add(layers.Flatten())
        self.model.add(layers.Input(shape=((int)(prod(inputShape)))))
        
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(dropout))
        
        #Decoder
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dropout(dropout))
        
        # self.model.add(layers.Dense(prod(inputShape), activation='relu'))
        # self.model.add(layers.Reshape(inputShape))        
        self.model.add(layers.Dense(prod(inputShape), activation=None))
        
class AECNN(NeuralNetwork):
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        NeuralNetwork.__init__(self, inputShape)
        
        self.modelName = "AECNN"
        
        dropout=0.25
        self.model = Sequential()
        
        #CNN encoder
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same', input_shape=inputShape))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.MaxPooling2D(2,padding='same'))
        self.model.add(layers.Dropout(dropout))
        
        #Flatten
        shape = self.model.output_shape[1:]
        shapeFlat = np.prod(shape)
        self.model.add(layers.Flatten())
        
        
        #Encoder
        self.model.add(layers.Dense(shapeFlat, activation='relu'))
        self.model.add(layers.Dropout(dropout))
        #self.model.add(layers.Dense(512, activation='relu'))
        #self.model.add(layers.Dropout(dropout))
        
        self.model.add(layers.Dense(258, activation='relu'))
        self.model.add(layers.Dropout(dropout))
        
        #Decoder
        #self.model.add(layers.Dense(512, activation='relu'))
        #self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Dense(shapeFlat, activation='relu'))
        self.model.add(layers.Dropout(dropout))
        
        #CNN decoder
        self.model.add(layers.Reshape(shape)) 
        
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.UpSampling2D(2))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Conv2D(8,5,activation='relu',padding='same'))
        self.model.add(layers.UpSampling2D(2))

        
        #Output
        self.model.add(layers.Conv2D(self.inputShape[-1],3,activation=None,padding='same'))  
        
class VAECNN(VAE_MODEL):
    # https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/
    def __init__(self, inputShape = (28,28,3)):
        super().__init__()
        VAE_MODEL.__init__(self, inputShape)
        
        
        self.modelName = "VAECNN"
        dropout=0.2
        
        #Loss metrics
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        
        
        #Encoder
        self.vae_input = Input(shape=inputShape, name="input_layer")
                
        self.encoderCNN = Sequential()
        self.encoderCNN.add(layers.Conv2D(32, (2, 2), strides = 1, padding = 'same', input_shape = inputShape))
        self.encoderCNN.add(layers.MaxPooling2D(2,padding='same'))
        self.encoderCNN.add(layers.BatchNormalization())
        self.encoderCNN.add(layers.ReLU())
        self.encoderCNN.add(layers.Dropout(dropout))
        self.encoderCNN.add(layers.Conv2D(64, (2, 2), strides = 1, padding = 'same'))
        self.encoderCNN.add(layers.MaxPooling2D(2,padding='same'))
        self.encoderCNN.add(layers.BatchNormalization())
        self.encoderCNN.add(layers.ReLU())
        self.encoderCNN.add(layers.Dropout(dropout))
        self.encoderCNN.add(layers.Conv2D(128, (2, 2), strides = 1, padding = 'same'))
        self.encoderCNN.add(layers.MaxPooling2D(2,padding='same'))
        self.encoderCNN.add(layers.BatchNormalization())
        self.encoderCNN.add(layers.ReLU())
        self.encoderCNN.add(layers.Dropout(dropout))
       
        #Flatten
        shape = self.encoderCNN.output_shape[1:]
        shapeFlat = np.prod(shape)
        self.encoderCNN.add(layers.Flatten())
        #self.encoderCNN.add(layers.Dense(32, activation="relu"))
        
        
        #Distribution calulcation space
        latent_space_dim = 2
        self.encoderCNN_output = self.encoderCNN(self.vae_input)
        self.encoder_mean = layers.Dense(units=latent_space_dim, name="encoder_mu")(self.encoderCNN_output)
        self.encoder_log_variance = layers.Dense(units=latent_space_dim, name="encoder_log_variance")(self.encoderCNN_output)
        self.encoder_distribution_output = self.Sampling()([self.encoder_mean, self.encoder_log_variance])
        
        self.encoder = Model(self.vae_input, [self.encoder_mean, self.encoder_log_variance, self.encoder_distribution_output], name='encoder_model')
        
        
        #Latent layer
        self.input_latent = layers.Input(shape=(latent_space_dim,), name="latent_input")
        self.ouput_latent_dense1 = layers.Dense(shapeFlat, activation="relu")(self.input_latent)
        self.latent = Model(self.input_latent, self.ouput_latent_dense1, name="latent_space")
        

        
        #Decoder
        self.decoder_input = Input(shape=(shapeFlat,), name="decoder_input")

        self.decoderCNN = Sequential()
        self.decoderCNN.add(layers.Reshape(shape))
        self.decoderCNN.add(layers.Conv2DTranspose(128, (2, 2), strides = 1, padding = 'same'))
        self.decoderCNN.add(layers.UpSampling2D(2))
        self.decoderCNN.add(layers.BatchNormalization())
        self.decoderCNN.add(layers.ReLU())
        self.decoderCNN.add(layers.Dropout(dropout))
        self.decoderCNN.add(layers.Conv2DTranspose(64, (2, 2), strides = 1, padding = 'same'))
        self.decoderCNN.add(layers.UpSampling2D(2))
        self.decoderCNN.add(layers.BatchNormalization())
        self.decoderCNN.add(layers.ReLU())
        self.decoderCNN.add(layers.Dropout(dropout))
        self.decoderCNN.add(layers.Conv2DTranspose(32, (2, 2), strides = 1, padding = 'same'))
        self.decoderCNN.add(layers.UpSampling2D(2))
        self.decoderCNN.add(layers.BatchNormalization())
        self.decoderCNN.add(layers.ReLU())
    
        self.decoderCNN.add(layers.Conv2DTranspose(self.inputShape[-1], (1, 1), strides = 1, activation = 'sigmoid', padding = 'same'))
        
        self.decoderOutput = self.decoderCNN(self.decoder_input)
        self.decoder = Model(self.decoder_input, self.decoderOutput, name = "decoder_model")
        
        #VAE model creation
        self.encoder_output = self.encoder(self.vae_input)
        self.latent_output = self.latent(self.encoder_output[2])
        self.vae_output = self.decoder(self.latent_output)
        self.model = Model(self.vae_input, self.vae_output, name='VAE')

        #self.model = VAE_MODEL(self.encoder, self.latent, self.decoder, name="VAE")
        #self.model.build((None,self.inputShape[0],self.inputShape[1],self.inputShape[2]))

        losses = self.loss_vae(self.vae_input, self.vae_output, self.encoder_mean, self.encoder_log_variance, self.inputShape, self.loss)
        self.model.add_loss(losses[0])
        self.model.add_metric(losses[1], name="kl_loss")
        self.add_metric(losses[2], name="reconstruction_loss")
        
    


def score(target,prediction):
    score = 0
    for i in range(len(target)):
        diff = target[i] - prediction[i]
        score += sqrt(diff ** 2) 
        score = 10**-score
        score = score*10
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
    

    













