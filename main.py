# -*- coding: utf-8 -*-
"""
Game prototype using image detection

Credit: Law Tech Productions, Adrien Dorise
February 2023
"""


if __name__ == "__main__":
	from dragonflai.config.ML_config import *
	from dragonflai.config.NN_config import *
	from dragonflai.config.data_config import *
	from dragonflai.experiment_generative import *
	from dragonflai.experiment_clustering import *
 
	experiment = Experiment_Clustering(NN_model,
			train_path,
			val_path,
			test_path,
			visu_path,
			input_shape=input_shape,
			num_epoch=num_epoch,
			batch_size=batch_size,
			learning_rate=lr,
			weight_decay=wd,
			optimizer=optimizer,
			criterion=crit,
			scaler=scaler,
			use_scheduler=use_scheduler,
			nb_workers=nb_workers)

	experiment.model.printArchitecture((1,3,input_shape[0],input_shape[1]))
	#experiment = Experiment_Generative.load("models/tmp/experiment")
	experiment.fit()
	results = experiment.predict()
	print(results)
	experiment.visualise()