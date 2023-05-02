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
xSize = 224
ySize = xSize
colorMode = 'rgb' #'rgb', 'monochrome'
flatInput = False

epoch = 1000
batch_size = 32
optimizer = 'adam' #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
loss = 'mse' #https://www.tensorflow.org/api_docs/python/tf/keras/losses
learningRate = 0.00001

trainFolder = 'OIDv4_ToolKit/OID/Dataset/train'
testFolder = 'OIDv4_ToolKit/OID/Dataset/test'
validationFolder = 'OIDv4_ToolKit/OID/Dataset/validation'

resumeTraining = True




if(colorMode == 'rgb'):
    inputShape = (xSize,ySize,3)
else:
    inputShape = (xSize,ySize,1)

model = nn.ConvolutionalAutoEncoders2(inputShape)





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
    model.loadModel("models/temp1")
    history = pd.DataFrame()
    model.build(optimizer = optimizer, loss = loss, lr = learningRate)
    model, history = pr.resumeTraining(model, history, train, validation, learningRate, epoch, batch_size)

saveFile = 'models/AE'
model.saveModel(saveFile, history, overwrite=False)
model.saveModel('models/temp', history, overwrite=True)



# model = nn.AECNN(inputShape)
# model.build(optimizer = optimizer, loss = loss, lr = learningRate)
# model.loadModel("models/testAECNN1")

# modeltl = tl.tlModel('VGG16')




#!!!!!TESTING!!!!!
print("Plot")

fileTest = 'OIDv4_ToolKit/OID/Dataset/train/Banana/7a270f199e78c912.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

fileTest = 'OIDv4_ToolKit/OID/Dataset/test/Banana/eeb93d366d6c69e7.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

fileTest = 'OIDv4_ToolKit/OID/Dataset/test/Banana/694f86.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

fileTest = 'Dataset_example/test/voiture.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

fileTest = 'Dataset_example/test/Cavendish-Banana-s.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

fileTest = 'Dataset_example/test/controller.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

fileTest = 'Dataset_example/test/tass.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

fileTest = 'Dataset_example/train/00ac03de349a3c5b.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)

fileTest = 'Dataset_example/train/00d843af60eecf7c.jpg'
pr.testModel(model, fileTest, inputShape, flatInput)
# modeltl.predict(fileTest)


