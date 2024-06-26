"""
 Parameters for machine learning applications
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - November 2023
"""

import sklearn.tree as tree
from dragonflai.model.machineLearning import Regressor


availableModels = ["tweedie", "bayesLinear", "SGD", "SVM", "KNN", "tree", "forest", "AdaBoost", "GBoost"]

tweedie_parameters = {"power": [0,1,1.5,2,3], 
                        "alpha": [0,0.25,0.5,0.75,1,2.5,5,10], 
                        "link": ["auto"]}
bayesLinear_parameters = {"alpha_1": [1e-05,1e-06,1e-07], 
                            "alpha_2": [1e-05,1e-06,1e-07], 
                            "lambda_1": [1e-05,1e-06,1e-07], 
                            "lambda_2": [1e-05,1e-06,1e-07]} 
SGD_parameters = {'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha' : [0.001,0.0005,0.0001],
                    'learning_rate': ['constant', 'optimal', 'invscaling']}
SVM_parameters = {'kernel' : ('linear', 'rbf', 'sigmoid'),
                    'C' : [1,5,10],
                    'epsilon': [0.01,0.1,0.5,1],
                    'coef0' : [0.01,10,0.5],
                    'gamma' : ('auto','scale')}
KNN_parameters = {"n_neighbors": range(1,10),
                    "p": [1,2]}
tree_parameters = {"criterion": ["squared_error", "absolute_error", "friedman_mse"],
                    "max_depth": [None, 100,1000, 5000]}
forest_parameters = {"n_estimators": [10,50,100],
                    "criterion": ["squared_error", "absolute_error", "friedman_mse"],
                    "max_depth": [None, 100,1000]}
adaBoost_parameters = {"estimators": [None, tree.DecisionTreeRegressor(max_depth=50)],
                        "n_estimators": [10,50,100],
                        "learning_rate": [0.85,1,1.25]}
GBoost_parameters = {"loss": ["squared_error", "absolute_error", "huber"],
                        "n_estimators": [10,50,100],
                        "learning_rate": [0.85,1,1.25]}

parametersModel = [tweedie_parameters, bayesLinear_parameters, SGD_parameters, SVM_parameters, KNN_parameters, tree_parameters, forest_parameters, adaBoost_parameters, GBoost_parameters]

ML_model = Regressor("SVM")
    