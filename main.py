# -*- coding: utf-8 -*-
"""
Game prototype using image detection

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""

import processing as proc
import AutoEncoders as AE
import numpy as np
import cv2


#Dynamic memory
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# session.close()


#Parameters
xSize = 128
ySize = 128
colorMode = 'rgb' #'rgb', 'monochrome'

epoch = 1000
batch_size = 32
optimizer = 'adam' #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
loss = 'msle' #https://www.tensorflow.org/api_docs/python/tf/keras/losses
learningRate = 0.00001

trainFolder = 'OIDv4_ToolKit/OID/Dataset/train'
testFolder = 'OIDv4_ToolKit/OID/Dataset/test'
validationFolder = 'OIDv4_ToolKit/OID/Dataset/validation'

resumeTraining = True



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


train,train_flat = proc.loadFolder(trainFolder + "/Banana",inputShape)
validation, validation_flat = proc.loadFolder(validationFolder + "/Banana", inputShape)
test, test_flat = proc.loadFolder(testFolder + "/Banana", inputShape)
print(f"Data info: train: {train.shape[0]} / validation: {validation.shape[0]}/ test: {test.shape[0]}")




#!!!!!TRAINING!!!!!
if not resumeTraining:
    model, history = proc.trainNew(train_flat, validation_flat, inputShape, optimizer, loss, learningRate, epoch, batch_size)
else:
    model, history = proc.resumeTraining(model, history, train_flat, validation_flat, learningRate, epoch, batch_size)

saveFile = 'models/AE'
model.saveModel(saveFile)
# model.loadModel("models/AE8")


#!!!!!TESTING!!!!!
print("Plot")

fileTest = 'OIDv4_ToolKit/OID/Dataset/train/Banana/06a804f6a15ce815.jpg'
proc.testModel(model, fileTest, inputShape)

fileTest = 'OIDv4_ToolKit/OID/Dataset/test/Banana/eeb93d366d6c69e7.jpg'
proc.testModel(model, fileTest, inputShape)

fileTest = 'OIDv4_ToolKit/OID/Dataset/test/Banana/694f86.jpg'
proc.testModel(model, fileTest, inputShape)





