"""
 Data related (paths, preprocessing...) parameters
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - November 2023
"""

from sklearn.preprocessing import MinMaxScaler


train_path = r"datasets/Banana/train/"
val_path = r"datasets/examples/Banana/validation/"
test_path = r"datasets/examples/Banana/test/"
visu_path = r"datasets/examples/Banana/test/"
save_path = r"models/tmp/dummy_experiment/"

input_shape = (128,128)
colorMode = 'rgb' #'rgb', 'monochrome'
scaler = None
nb_workers = 0

