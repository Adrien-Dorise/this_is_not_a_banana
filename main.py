# -*- coding: utf-8 -*-
"""
Game prototype using image detection

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""


import AutoEncoders as AE
import numpy as np

#Parameters
xSize = 256
ySize = 256
colors = 3
colorMode = 'rgb' #'rgb', 'grayscale

epoch = 100
batch_size = 16



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
                                                 batch_size = batch_size,
                                                 class_mode = 'input',
                                                 color_mode=colorMode,
                                                 )

test_set = test_datagen.flow_from_directory(validationFolder,
                                            target_size = (xSize, ySize),
                                            batch_size = batch_size,
                                            class_mode = 'input',
                                            color_mode=colorMode)

#!!!!!NEURAL NETWORK!!!!!

a = AE.AutoEncoders([xSize,ySize], colors)
a.build()


a.printInfos()
a.fit(training_set,test_set,epochs=epoch)

saveFile = 'models/AE'
a.saveModel(saveFile)
# a.loadModel("models/AE2")

# Encode and decode some digits
# Note that we take them from the *test* set
# imgTEST = a.predict(test_set)

#!!!!!TESTING!!!!!
print("Plot")
fileTest = 'OIDv4_ToolKit/OID/Dataset/validation/Banana/356b1b9ddd7e3b22.jpg'
a.plotPrediction(fileTest)
fileTest = 'OIDv4_ToolKit/OID/Dataset/validation/Banana/694f86.jpg'
a.plotPrediction(fileTest)

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


