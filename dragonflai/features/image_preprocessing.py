"""
 Image preprocessing toolbox for data coming from the faceMask app
 Author: Julia Cohen - Adrien Dorise ({jcohen, adorise}@lrtechnologies.fr) - LR Technologies
 Created: March 2023
 Last updated: Adrien Dorise - November 2023
"""

import os

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



    
def flattenImage(image):
    """This function takes a (x,y,c) image and flatter into a 1D vector

    Args:
        image (_type_): Image to flatten

    Returns:
        np.array: 1d array of the image
    """
    flatImg = np.array(image).flatten()
    return flatImg.reshape(-1,len(flatImg))[0]

def reverseFlatten(image,imgShape):
    """Reverse an image 

    Args:
        image (_type_): _description_
        imgShape (_type_): _description_

    Returns:
        _type_: _description_
    """
    return image.reshape(imgShape[0],imgShape[1],imgShape[2])    


class ImageRegressionDataset(Dataset):
    '''
    Class used to store the dataset handled by pytorch, for Image inputs
    '''
    def __init__(self, data_folder, shape=(256,256)):
        '''
        Dataset class constructor.
        Parameters
        ----------
        data_folder: 
            folder with images to use as features and targets. These two folders must appear in the hierarchy below data_folder.
        shape:
            X and Y dimension of the images. They will be reshape into this parameters. Default is (256,256).
        Returns
        ----------
        None
        '''
        
        self.feature_path = data_folder + "features/"
        self.target_path = data_folder + "targets/"
        self.features = [cv2.imread(self.feature_path + img) for img in os.listdir(self.feature_path)]
        self.targets = [cv2.imread(self.target_path + img) for img in os.listdir(self.target_path)]
        if len(self.features) == 0 or (self.targets) == 0:
            raise Exception("No img file found")
        if(len(self.features) != len(self.targets)):
            raise Exception("Not the same number of features and targets! Must be an error in your dataset")

        self.samples_number = len(self.features)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # convert from [0, 255] to [0.0, 0.1]
            ])

        self.shape = shape
        self.crop_fct = None
        self.detector = None
                
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Find the cap object holding frame of correct index
        feature = self.features[idx]
        target = self.targets[idx]
        assert feature is not None, \
            f"Feature {idx} not loaded ({self.feature_path[idx]})."
        assert target is not None, \
            f"Target {idx} not loaded ({self.target_path[idx]})."        
                        
        feature = cv2.resize(feature, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
        feature = self.to_tensor(feature)
        target = cv2.resize(target, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
        target = self.to_tensor(target)
        
        return feature, target

    def __del__(self):
        for feat in self.features:
           del feat
        for targ in self.targets:
            del targ


def image_collate_fn(data):
    """
    Custom collate function that returns the images and targets 
    from a batch in the form of 2 tuples, instead of a single sequence.

    Parameters
    ----------
    data: list
        Sequence of (image, target) produced by the dataloader.
    Returns
    ----------
    imgs: list
        List of images (size as the batch size of the dataloader).
    targets: list
        List of 
    """
    imgs, targets = zip(*data)  # tuples
    return torch.stack(imgs, 0), torch.stack(targets, 0)

def img_loader(folder_path, 
               shape=(256,256),
               batch_size=16, 
               shuffle=True, 
               num_workers=0,):
    """Create Pytorch DataLoader object from array dataset
    
    Args:
        features (array of size (inputs, features)): set containing the features values
        target (set of size (inputs, targets)): Set containing the targets values
        batch_size (int, optional): batch_size used for DataLoader. Defaults to 16.
        shuffle (bool, optional): True to shuffle the dataset. Defaults to True
    Returns:
        dataset: Pytorch DataLoader object
    """

    dataset = ImageRegressionDataset(folder_path, 
                                    shape=shape)
        
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers, 
                        collate_fn=image_collate_fn,
                        shuffle=shuffle)
    return loader
