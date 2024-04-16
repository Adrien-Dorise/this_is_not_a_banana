"""
This script contains all functions related to the scoring function of the game.
Author: Adrien Dorise (lawtechprod@hotmail.com) - Law Tech Productions
Created: April 2024
Last updated: Adrien Dorise - April 2024

"""

import matplotlib.pyplot as plt
import dragonflai.features.image_preprocessing as pr
import cv2
import dragonflai.postprocess.score as sc

def plotImage(image):
    plt.imshow(image[:,:,::-1])
    plt.show()
    

def save_results(target, prediction, save_path):   
    
    target = target.transpose(1,2,0)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    prediction = prediction.transpose(1,2,0)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)        
        

    score = sc.score(target, prediction)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Prediction comparison\nscore=' + str(int(score)))

    ax1.imshow(target)
    ax2.imshow(prediction)
    plt.savefig(save_path)


if __name__ == "__main__":
    pass