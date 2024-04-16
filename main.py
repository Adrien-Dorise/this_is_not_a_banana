# -*- coding: utf-8 -*-
"""
Game prototype using image detection

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""

import sys
sys.path.append("scripts/")
import processing as pr
import neuralNetwork as nn
import numpy as np
import cv2
import pandas as pd
import transferLearning as tl
import keras.metrics as metrics
from sklearn.utils import shuffle


#Dynamic memory
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# session.close()


#Parameters
xSize = 128
ySize = xSize
colorMode = 'rgb' #'rgb', 'monochrome'
flatInput = False

epoch = 250
batch_size = 32
optimizer = 'adam' #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
loss = 'binary_crossentropy' #https://www.tensorflow.org/api_docs/python/tf/keras/losses
learningRate = 0.001

trainFolder = 'OIDv4_ToolKit/OID/Dataset/train'
testFolder = 'OIDv4_ToolKit/OID/Dataset/test'
validationFolder = 'OIDv4_ToolKit/OID/Dataset/validation'

resumeTraining = False




if(colorMode == 'rgb'):
    inputShape = (xSize,ySize,3)
else:
    inputShape = (xSize,ySize,1)

model = nn.VAECNN(inputShape)

#if(model.modelName == "VAECNN"):
#    loss = None


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


train2D,train_flat = pr.loadFolder(trainFolder + "/Banana",inputShape)
validation2D, validation_flat = pr.loadFolder(validationFolder + "/Banana", inputShape)
test2D, test_flat = pr.loadFolder(testFolder + "/Banana", inputShape)
print(f"Data info: train: {train2D.shape[0]} / validation: {validation2D.shape[0]}/ test: {test2D.shape[0]}")

if not flatInput:
    train = shuffle(train2D)
    validation = validation2D
    test = test2D
else:
    train = shuffle(train_flat)
    validation = validation_flat
    test = test_flat




#!!!!!TRAINING!!!!!
if not resumeTraining:
    model, history = pr.trainNew(model, train, validation, optimizer, loss, learningRate, epoch, batch_size)
else:    
    history = model.load_weights("models/temp1")
    model.build(optimizer = optimizer, loss = loss, lr = learningRate)
    model, history = pr.resumeTraining(model, history, train, validation, learningRate, epoch, batch_size)

saveFile = 'models/' + model.name
model.save_weights(saveFile, history, overwrite=False)
model.save_weights('models/temp', history, overwrite=True)




# model = nn.AECNN(inputShape)
# model.build(optimizer = optimizer, loss = loss, lr = learningRate)
# model.loadModel("models/testAECNN1")

# modeltl = tl.tlModel('VGG16')




#!!!!!TESTING!!!!!
print("Plot")
#model.load_weights("models/temp1")

print("\nbanana train")
fileTest = 'OIDv4_ToolKit/OID/Dataset/train/Banana/7a270f199e78c912.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

print("\npear-banana test")
fileTest = 'OIDv4_ToolKit/OID/Dataset/test/Banana/eeb93d366d6c69e7.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

print("\nmeme")
fileTest = 'OIDv4_ToolKit/OID/Dataset/test/Banana/694f86.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

print("\ncar")
fileTest = 'Dataset_example/test/voiture.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

print("\nbanana plate test")
fileTest = 'Dataset_example/test/Cavendish-Banana-s.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

print("\ncontroller")
fileTest = 'Dataset_example/test/controller.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

print("\nmug")
fileTest = 'Dataset_example/test/tass.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

print("\nbanana train")
fileTest = 'Dataset_example/train/00ac03de349a3c5b.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

print("\nbanana train")
fileTest = 'Dataset_example/train/00d843af60eecf7c.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)



import matplotlib.pyplot as plt


def plot_latent_space(vae, n=1, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 256
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_latent = vae.latent.predict(z_sample)
            x_decoded = vae.decoder.predict(x_latent)
            digit = x_decoded[0].reshape(digit_size, digit_size,3)

    plt.imshow(digit)
    plt.show()


plot_latent_space(model.model)



from numba import cuda
cuda.select_device(0)
cuda.close()
