import numpy as np
from datasets import load_dataset
from pathlib import Path
from torch.utils.data import DataLoader
from svm import SVM
from itertools import product

import pdb

def get_dataset(ds_name='emnist', data_dir='../datasets', batch_size=16):
	"""
	downloads the dataset and returns a dataloader for batching

	:param ds_name: name of the dataset. check huggingface for valid dataset names
	:param data_dir: the directory to download and store the data
	:param batch_size: the batch size for the classifier to train on

	:return: two pytorch dataloaders for the training and testing data
	"""
	# create path for dataset if it doesn't exist
	Path(data_dir).mkdir(parents=True, exist_ok=True)
	
	# download or read in dataset
	ds = load_dataset(
		'mnist',
		data_dir=data_dir,
	).with_format('numpy')

	return  (
		DataLoader(ds['train'], batch_size=batch_size),
		DataLoader(ds['test'], batch_size=batch_size),
	)

def train():
	""" instantiate and train an SVM classifier """

	# TODO make these arguments
	# hard-coded parameters. should c
	C = 10
	dim = 784
	lr=1e-4
	model = SVM(C, dim, lr=lr)
	batch_size=2**12
	num_epochs=10

	# get dataset
	train_data, test_data = get_dataset(batch_size=batch_size)

	# outer training loop is for each epoch
	for epoch in range(num_epochs):
		
		# inner loop iterates through the whole dataset
		for batch in train_data:

			# split into data and labels. 
			data, labels = (batch['image'].numpy(), batch['label'].numpy())
				
			# compute loss
			loss, grad = model.loss(data, labels)
			print(f'loss: {loss}\n')

			# optimizer step
			model.optim_step(grad)
if __name__ == '__main__':
	train()
