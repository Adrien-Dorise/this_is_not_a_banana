# -*- coding: utf-8 -*-
"""
Convolutional network prototype for a recognition mobile game

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""

# Verify GPU is available
# https://www.tensorflow.org/install/pip#hardware_requirements
# import tensorflow as tf
# tf.test.gpu_device_name()

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Parameters
xSize = 16
ySize = 16

#!!!!!NETWORK CREATION!!!!!

#Initialising the network
classifier = Sequential()

#First Convolution layer
classifier.add(Conv2D(32, (3,3), input_shape = (xSize,ySize,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#2nd convolution layer
classifier.add(Conv2D(32, (3,3), input_shape = (xSize,ySize,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
classifier.add(Flatten())

#ANN
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Configuring network for training
classifier.compile(optimizer='rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])




#!!!!!IMAGE PROCESSING!!!!!
#Using tensorflow dataset to get openImage (don't work with python 3.9 right now)
#NOTE: tfds not working alongside tensorflow-gpu
#https://www.tensorflow.org/datasets/overview
# import tensorflow_datasets as tfds
# dataset = tfds.load('open_images/v7', split='train')

#Openimage
#https://storage.googleapis.com/openimages/web/download_v7.html#download-tfds

#Downloading images from open image with OIDv4 first (see related folder)
#https://github.com/EscVM/OIDv4_ToolKit

trainFolder = 'OIDv4_ToolKit/OID/Dataset/train'
testFolder = 'OIDv4_ToolKit/OID/Dataset/test'
validationFolder = 'OIDv4_ToolKit/OID/Dataset/validation'


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(trainFolder,
                                                 target_size = (xSize, ySize),
                                                 batch_size = 16,
                                                 class_mode = 'binary',
                                                 color_mode='rgb',
                                                 )

test_set = test_datagen.flow_from_directory(testFolder,
                                            target_size = (xSize, ySize),
                                            batch_size = 16,
                                            class_mode = 'binary',
                                            color_mode='rgb')

#!!!!!TRAINING!!!!!
classifier.fit(training_set,
               steps_per_epoch = None, # This number is done by dividing total number of pictures in train set by batch size (here 8000/32)
               epochs = 16,
               validation_data = test_set,
               validation_steps = None)

#!!!!!TESTING!!!!!
import keras.utils as image
fileTest = 'OIDv4_ToolKit/OID/Dataset/validation/Banana/356b1b9ddd7e3b22.jpg'
fileTest = 'OIDv4_ToolKit/OID/Dataset/validation/Banana/694f86.jpg'
test_image = image.load_img(fileTest, target_size = (xSize, ySize))
test_image = image.img_to_array(test_image)
test_image = test_image/255.0
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
probabResult = result[0][0]
print(probabResult)
    
    
    
    
    
    
    
    
    
    

