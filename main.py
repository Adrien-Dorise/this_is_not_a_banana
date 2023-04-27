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
import transferLearning as tl
import keras.metrics as metrics

#Dynamic memory
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# session.close()


#Parameters
xSize = 224
ySize = 224
colorMode = 'rgb' #'rgb', 'monochrome'
flatInput = False

epoch = 500
batch_size = 64
optimizer = 'adam' #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
loss = 'mae' #https://www.tensorflow.org/api_docs/python/tf/keras/losses
learningRate = 0.0001

trainFolder = 'OIDv4_ToolKit/OID/Dataset/train'
testFolder = 'OIDv4_ToolKit/OID/Dataset/test'
validationFolder = 'OIDv4_ToolKit/OID/Dataset/validation'

resumeTraining = False




if(colorMode == 'rgb'):
    inputShape = (xSize,ySize,3)
else:
    inputShape = (xSize,ySize,1)


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
    train = train2D
    validation = validation2D
    test = test2D
else:
    train = train_flat
    validation = validation_flat
    test = test_flat
    

#!!!!!TRAINING!!!!!
if not resumeTraining:
    model, history = pr.trainNew(train, validation, inputShape, optimizer, loss, learningRate, epoch, batch_size)
else:
    model, history = pr.resumeTraining(model, history, train, validation, learningRate, epoch, batch_size)

saveFile = 'models/AE'
model.saveModel(saveFile, history, overwrite=False)

# model = nn.AECNN(inputShape)
# model.build(optimizer = optimizer, loss = loss, lr = learningRate)
# model.loadModel("models/testAECNN1")

# modeltl = tl.tlModel('VGG16')




#!!!!!TESTING!!!!!
print("Plot")

fileTest = 'OIDv4_ToolKit/OID/Dataset/train/Banana/06a804f6a15ce815.jpg'
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


