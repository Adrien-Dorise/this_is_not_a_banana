"""
Processing toolbox for image prediction

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""

import neuralNetwork as nn

from os import listdir
from os.path import isdir
from matplotlib.pyplot import imshow, show
from cv2 import imread, resize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
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
"""


def loadImage(filePath, imgShape):
            img=imread(filePath)
            img=resize(img,(imgShape[0],imgShape[1]))
            img = img/255.0
            if(imgShape[2] == 1):
                img = np.mean(img, -1)
            return img
            
def loadFolder(folderPath, imgShape):
    x = []
    x_flat = []
    for img in listdir(folderPath):
        filePath = folderPath + "/" + img
        if(not isdir(filePath)):
            imgTemp = loadImage(filePath, imgShape) 
            x.append(imgTemp)
            x_flat.append(imgTemp.flatten())
    x = np.array(x)
    # x = x.reshape(-1,xSize,ySize,x.shape[-1])
    x_flat = np.array(x_flat)
    # x_flat = x_flat.reshape(-1,len(x_flat)).transpose()
    return x, x_flat

def plotImage(image):
    plt.imshow(image[:,:,::-1])
    plt.show()
    
def plotFlatImage(image, imgShape):
    img = reverseFlatten(image, imgShape)
    plotImage(img)
    
def flattenImage(image):
    flatImg = np.array(image).flatten()
    return flatImg.reshape(-1,len(flatImg))

def reverseFlatten(image,imgShape):
    return image.reshape(imgShape[0],imgShape[1],imgShape[2])    


def plotTraining(title,history):
    fig = plt.figure(figsize=(6,4),dpi=250)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.plot(history["loss"], label = "Train")
    plt.plot(history["val_loss"], label = "Validation")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()
    
    
def trainNew(model,trainSet, validSet, optimizer, loss, lr, epoch, batch_size):
    history_pd = pd.DataFrame()
    model.build(optimizer = optimizer, loss = loss, lr = lr)
    model.printInfos()
    history = model.fit(trainSet,validSet,epochs=epoch, batch_size = batch_size)
    history_pd = history_pd.append(pd.DataFrame(history.history), ignore_index=True)
    plotTraining(model.modelName, history_pd)
    return model, history_pd
    
def resumeTraining(model, history, trainSet, validSet, lr, epoch, batch_size):
    model.model.optimizer.learning_rate = lr
    hist = model.fit(trainSet,validSet,epochs=epoch, batch_size = batch_size)
    history = history.append(pd.DataFrame(hist.history), ignore_index=True)
    plotTraining(model.modelName, history)
    return model, history

def testModel(model, fileTestPath, inputShape, flat=False):
    testImg = loadImage(fileTestPath, inputShape)
    if(flat):    
        testImg = flattenImage(testImg)
        plotFlatImage(testImg[0], inputShape)
        predict = model.predict(testImg)
        plotFlatImage(predict, inputShape)
    else:
        plotImage(testImg)
        test = testImg.reshape(-1, inputShape[0], inputShape[1], inputShape[2])
        predict = model.predict(test)
        predictImg = predict[0,:,:,:]
        plotImage(predictImg)
        testImg = flattenImage(testImg)
        predict = flattenImage(predict)
    score = nn.score(testImg[0], predict[0])
    print("score: " + str(score))
    return score
    
    
    
    
    
    
    
    
    
    
    


