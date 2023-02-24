"""
Pre-processing toolbox for image prediction

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""

from os import listdir
from os.path import isdir
from matplotlib.pyplot import imshow, show
from cv2 import imread, resize
from numpy import newaxis, array, mean
import matplotlib.pyplot as plt

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


def loadImage(filePath, xSize, ySize,colors):
            img=imread(filePath)
            img=resize(img,(xSize,ySize))
            img = img/255.0
            if(colors == 1):
                img = mean(img, -1)
            return img
            
def loadFolder(folderPath, xSize, ySize, colors):
    x = []
    x_flat = []
    for img in listdir(folderPath):
        filePath = folderPath + "/" + img
        if(not isdir(filePath)):
            imgTemp = loadImage(filePath, xSize, ySize,colors) 
            x.append(imgTemp)
            x_flat.append(imgTemp.flatten())
    x = array(x)
    # x = x.reshape(-1,xSize,ySize,x.shape[-1])
    x_flat = array(x_flat)
    # x_flat = x_flat.reshape(-1,len(x_flat)).transpose()
    return x, x_flat

def plotImage(image):
    plt.imshow(image[:,:,::-1])
    plt.show()
    
def plotFlatImage(image, xSize, ySize, colors):
    img = reverseFlatten(image, xSize, ySize, colors)
    plotImage(img)
    
def flattenImage(image):
    flatImg = array(image).flatten()
    return flatImg.reshape(-1,len(flatImg))

def reverseFlatten(image,xSize,ySize,colors):
    return image.reshape(xSize,ySize,colors)    


