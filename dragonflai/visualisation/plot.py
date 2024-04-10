"""
This script contains all functions related to the scoring function of the game.
Author: Adrien Dorise (lawtechprod@hotmail.com) - Law Tech Productions
Created: April 2024
Last updated: Adrien Dorise - April 2024

"""

import matplotlib.pyplot as plt
import dragonflai.features.image_preprocessing as pr

def plotImage(image):
    plt.imshow(image[:,:,::-1])
    plt.show()
    
def plotFlatImage(image, imgShape):
    img = pr.reverseFlatten(image, imgShape)
    plotImage(img)


if __name__ == "__main__":
    pass