# -*- coding: utf-8 -*-
"""
Game prototype using image detection

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""

import preprocessing as pre
import AutoEncoders as AE
import numpy as np
import cv2

#Parameters
xSize = 28
ySize = 28
colorMode = 'monochrome' #'rgb', 'monochrome'

epoch = 500
batch_size = 32





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


if(colorMode == 'rgb'):
    colors = 3
else:
    colors = 1


# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(#rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True,)

# test_datagen = ImageDataGenerator(rescale = 1./255)

# training_set = train_datagen.flow_from_directory(trainFolder,
#                                                  target_size = (xSize, ySize),
#                                                  batch_size = batch_size,
#                                                  class_mode = 'input',
#                                                  color_mode=colorMode,
#                                                  )

# test_set = test_datagen.flow_from_directory(validationFolder,
#                                             target_size = (xSize, ySize),
#                                             batch_size = batch_size,
#                                             class_mode = 'input',
#                                             color_mode=colorMode)


train,train_flat = pre.loadFolder(trainFolder + "/Banana", xSize, ySize, colors)
validation, validation_flat = pre.loadFolder(validationFolder + "/Banana", xSize, ySize, colors)
test, test_flat = pre.loadFolder(testFolder + "/Banana", xSize, ySize, colors)
print(f"Data info: train: {train.shape[0]} / validation: {validation.shape[0]}/ test: {test.shape[0]}")
pre.plotFlatImage(train_flat[0], xSize, ySize, colors)


#MNIST
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


#!!!!!NEURAL NETWORK!!!!!


a = AE.AE([xSize,ySize], colors)
a.build()


# a.printInfos()
history = a.fit(train_flat,test_flat,epochs=epoch, batch_size = batch_size)


import matplotlib.pyplot as plt
import pandas as pd
history_pd = pd.DataFrame(history.history)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.show()


saveFile = 'models/AE'
# a.saveModel(saveFile)
# a.loadModel("models/AE3")

# Encode and decode some digits
# Note that we take them from the *test* set
# imgTEST = a.predict(test_set)

#!!!!!TESTING!!!!!
print("Plot")

fileTest = 'OIDv4_ToolKit/OID/Dataset/test/Banana/b179fb7f7e447f77.jpg'
testImg = pre.loadImage(fileTest, xSize, ySize,colors)
testImg_flat = pre.flattenImage(testImg)
predict = a.predict(testImg_flat)
pre.plotFlatImage(testImg_flat[0], xSize, ySize, colors)
pre.plotFlatImage(predict, xSize, ySize, colors)
fileTest = 'OIDv4_ToolKit/OID/Dataset/test/Banana/eeb93d366d6c69e7.jpg'
testImg = pre.loadImage(fileTest, xSize, ySize,colors)
testImg_flat = pre.flattenImage(testImg)
predict = a.predict(testImg_flat)
pre.plotFlatImage(testImg_flat[0], xSize, ySize, colors)
pre.plotFlatImage(predict, xSize, ySize, colors)
fileTest = 'OIDv4_ToolKit/OID/Dataset/test/Banana/694f86.jpg'
testImg = pre.loadImage(fileTest, xSize, ySize,colors)
testImg_flat = pre.flattenImage(testImg)
predict = a.predict(testImg_flat)
pre.plotFlatImage(testImg_flat[0], xSize, ySize, colors)
pre.plotFlatImage(predict, xSize, ySize, colors)

#Show MNIST
testImg = x_test[5]
predict = a.predict(testImg.reshape(-1,784))
plt.imshow(testImg.reshape(28, 28))
plt.gray()
plt.show()
plt.imshow(predict.reshape(28, 28))
plt.gray()
plt.show()


# from scipy.linalg import norm
# from scipy import sum, average
# def normalize(arr):
#     rng = arr.max()-arr.min()
#     amin = arr.min()
#     return (arr-amin)*255/rng
# def compare_images(img1, img2):
#     # normalize to compensate for exposure difference, this may be unnecessary
#     # consider disabling it
#     img1 = normalize(img1)
#     img2 = normalize(img2)
#     # calculate the difference and its norms
#     diff = img1 - img2  # elementwise for scipy arrays
#     m_norm = sum(abs(diff))  # Manhattan norm
#     z_norm = norm(diff.ravel(), 0)  # Zero norm
#     return (m_norm, z_norm)
# print(compare_images(test_image,result))


