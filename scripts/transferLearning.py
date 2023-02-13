# -*- coding: utf-8 -*-
"""
Game prototype using image detection

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""

#Info from https://keras.io/api/applications/

from tensorflow.keras import applications
from tensorflow.keras import preprocessing
import numpy as np

class tlModel():
    def __init__(self, modelChoice = "ResNet50"):
        if modelChoice == "ResNet50":
            self.model = applications.ResNet50(weights='imagenet')
            self.name = modelChoice
        elif modelChoice == "VGG16":
            self.model = applications.VGG16(weights='imagenet')
            self.name = modelChoice
        else:
            print("WARNING: Transfler learning model choice unknown. ResNet50 taken as default")
            self.model = applications.ResNet50(weights='imagenet')
            self.name = modelChoice
        

    def predict(self, filePath):
        img_path = filePath
        img = preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if(self.name == 'ResNet50'):
            x = applications.resnet50.preprocess_input(x)
            
            preds = self.model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            print(self.name + ' prediction:', applications.resnet50.decode_predictions(preds, top=3)[0])
            
        elif(self.name == 'VGG16'):
            x = applications.vgg16.preprocess_input(x)
            preds = self.model.predict(x)
            print(self.name + ' prediction:', applications.vgg16.decode_predictions(preds, top=3)[0])
                
            
            