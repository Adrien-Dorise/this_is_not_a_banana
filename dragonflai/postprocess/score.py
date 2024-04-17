"""
This script contains all functions related to the scoring function of the game.
Author: Adrien Dorise (lawtechprod@hotmail.com) - Law Tech Productions
Created: April 2024
Last updated: Adrien Dorise - April 2024

"""

from math import sqrt
import dragonflai.features.image_preprocessing as pr
import numpy as np


def generation_score(target,prediction):
    score = 0
    target = pr.flattenImage(target)
    prediction = pr.flattenImage(prediction)
    for i in range(len(target)):
        diff = target[i] - prediction[i]
        score += sqrt(diff ** 2)
        #score = 10**-score
        #score = score*10
    return score

def clustering_score(centroid,encoded_features, lnorm=2):
    return np.linalg.norm(centroid-encoded_features,ord=lnorm) 



if __name__ == "__main__":
    pass