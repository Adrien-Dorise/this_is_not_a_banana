'''
This package references all neural network classes used in the application.
Author: Adrien Dorise - Edouard Villain ({adorise, evillain}@lrtechnologies.fr) - LR Technologies
Created: September 2023
Last updated: Adrien Dorise - September 2023

'''

import dragonflai.features.image_preprocessing as imgpr
from dragonflai.model.machineLearning import Regressor

import pickle
from sklearn.preprocessing import MinMaxScaler
import torch
from enum import Enum
import dragonflai.visualisation.plot as plot




class Experiment_Generative():
    """Regroup both a model and the data for a complete experiment.
    It can take into account both machine learning and neural network models
    It can fit with both tracker and full image datasets. Only the paths are stored, to avoid saving the whole dataset as Python variables. Data are loaded on the fly when needed (during fit or predict).

    Args:
            model (model class (such as Regressor/NeuralNetwork): algorithm selected
            train_path (string): Folder path containing the training samples
            val_path (string): Folder path containing the validation samples
            test_path (string): Folder path containing the testing samples
            visu_path (string): Folder path containing the samples used for result visualisation
            shape: X and Y dimension of the images. They will be reshape into this parameters. Default is (256,256).
            num_epoch (int, optional): Amount of epochs to perform during training. Defaults to 50.
            batch_size (int, optional): batch_size used for DataLoader. Defaults to 32.
            learning_rate (int, optional): learning_rate used during backpropagation. Defaults to 1e-03.
            weight_decay (int, optional): regularisation criterion. Defaults to 1e-03.
            optimizer (torch.nn, optional): Optimizer used during training for backpropagation. Defaults to torch.optim.Adam.
            criterion (torch.optim, optional): Criterion used during training for loss calculation. Defaults to torch.nn.L1Loss().
            scaler (sklearn.preprocessing, optional): Type of scaler used for data preprocessing. Put None if no scaler needed. Defaults to MinMaxScaler().
            nb_workers (int, optional):  How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default to 0. Defaults to 0.
    """
    def __init__(self, model,
                train_path,
                val_path,
                test_path,
                visu_path,
                input_shape=(256,256),
                num_epoch=50,
                batch_size=32,
                learning_rate=1e-03,
                weight_decay=1e-03,    
                optimizer=torch.optim.Adam,
                criterion=torch.nn.L1Loss(),
                use_scheduler=True,
                scaler = MinMaxScaler(),
                nb_workers=0):
        
        #Model parameters  
        self.model = model
        self.no_batch = False
        self.scheduler = use_scheduler 
        self.loss_indicators = 1

        #Path parameters
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.visu_path = visu_path
        
        #Features parameters
        self.input_shape = input_shape
        self.scaler = scaler
        self.shuffle = True
            
        #Training parameters
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.criterion = criterion
        self.nb_workers = nb_workers
         
    def fit(self):
        """Train the model using the data available in the train and validation folder path.
        """
        # !!! Data loading !!!
        train_set = imgpr.img_loader(self.train_path, shape=self.input_shape, shuffle=self.shuffle, 
                                     batch_size=self.batch_size, num_workers=self.nb_workers)
        val_set = imgpr.img_loader(self.val_path, shape=self.input_shape, shuffle=False,
                                   batch_size=self.batch_size, num_workers=self.nb_workers)
    

        #!!! Training!!! 
        losses_train, losses_val = self.model.fit(train_set,
        self.num_epoch, 
        criterion=self.criterion, 
        optimizer=self.optimizer,
        learning_rate=self.learning_rate,
        weight_decay=self.weight_decay, 
        valid_set=val_set,
        use_scheduler=self.scheduler,
        loss_indicators = self.loss_indicators)
        self.model.plotLoss(losses_train,losses_val)

        self.model.saveModel(f"models/tmp/model")
        self.save(f"models/tmp/experiment")
        

    def predict(self):          
        """Model prediction on the samples available in the test folder path
        """
        # !!! Data loading !!!
        test_set = imgpr.img_loader(self.test_path, shape=self.input_shape, shuffle=False,
                                    batch_size=self.batch_size,num_workers=self.nb_workers)
        score, prediction, _ = self.model.predict(test_set,self.criterion)



    def visualise(self):
        """Visualisation of the pictures in the visualisation set.
        The input + predicted images are both shown.
        """
        # !!! Data loading !!!
        visu_set = imgpr.img_loader(self.visu_path, shape=self.input_shape, shuffle=False,
                                    batch_size=self.batch_size, num_workers=self.nb_workers)
        
        score, pred, (feature, target) = self.model.predict(visu_set,self.criterion)

        for i in range(len(target)):
            plot.save_results(target[i], pred[i], f"output/tmp/img{i}.png")



    def save(self, filename):
        """Save the whole experiment class as a pickle object.

        Args:
            filename (string): Path to save the experiment status
        """
        with open(filename, 'wb') as file:
            try:
                pickle.dump(self, file)
            except EOFError:
                raise Exception("Error in save experiment: Pickle was not able to save the file.")

    @classmethod
    def load(self, filename):
        """Load a pickle object to an Experiment class Python variable
        This is a class method. It means that a reference to the class is NOT necessary to call this method. Simply type <your_experiment = Experiment.load(filename)>

        Args:
            filename (string): Path to the pickle saved object.
        """
        with open(filename, 'rb') as file:
            try:
               return pickle.load(file)
            except EOFError:
                raise Exception("Error in load experiment: Pickle was not able to retrieve the file.")