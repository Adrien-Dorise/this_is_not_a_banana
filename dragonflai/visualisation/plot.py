"""
This script contains all functions related to the scoring function of the game.
Author: Adrien Dorise (lawtechprod@hotmail.com) - Law Tech Productions
Created: April 2024
Last updated: Adrien Dorise - April 2024

"""

import matplotlib.pyplot as plt
import cv2
from math import sqrt
import numpy as np


def plotImage(image):
    plt.imshow(image[:,:,::-1])
    plt.show()
    
def get_closest_multiply(integer):
    a = int(sqrt(integer)) + 1
    while (integer % a != 0):
        a -= 1
    b = integer//a
    return a, b


def plot_generation(target, prediction, score, save_path):   
    target = target.transpose(1,2,0)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    prediction = prediction.transpose(1,2,0)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)        
        
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Generation comparison\nPixel-wise score=' + str(int(score)))

    ax1.imshow(target)
    ax2.imshow(prediction)
    plt.savefig(save_path)

def visualise_conv_filters(encoder, save_path):
    filters = encoder[-1].layer[0].weight.detach().cpu()
    filters = (filters - filters.min()) / (filters.max() - filters.min())
    filters = filters.reshape(-1,np.shape(filters)[-2],np.shape(filters)[-1])
    num_filters = filters.size(0)
    y_size, x_size = get_closest_multiply(num_filters)
    fig, axes = plt.subplots(x_size, y_size, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(filters[idx], cmap="gray")
        ax.axis('off')
    plt.savefig(save_path)

def visualise_conv_result(conv_output, score, save_path):
    y_size, x_size = get_closest_multiply(np.shape(conv_output)[0])
    fig, axes = plt.subplots(x_size, y_size, figsize=(10, 10))
    fig.suptitle("Encoder filters output\nClustering score=" + str(round(score,4)))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(conv_output[idx], cmap='gray')  # Display each channel of the output
        ax.axis('off')
    plt.savefig(save_path)

if __name__ == "__main__":
    pass
