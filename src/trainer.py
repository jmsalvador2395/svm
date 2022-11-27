import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from .svm import SVM
from itertools import product
import datasets
from datasets import load_dataset
from tqdm.notebook import tqdm

import pdb

def get_dataloaders(ds_name='mnist', data_dir='./datasets', batch_size=16):
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
		ds_name,
		cache_dir = data_dir
	).with_format('numpy')

	return  (
		ds,
		DataLoader(ds['train'], batch_size=batch_size),
		DataLoader(ds['test'], batch_size=batch_size),
	)

def evaluate(model, data):
	N=len(data)
	loss=0
	acc=0
	# compute loss and accuracy on the test set
	for batch in data:

		# split into data and labels. 
		x, y = (batch['image'].numpy(), batch['label'].numpy())

		# compute loss
		loss_i, _, acc_i = model.loss(x, y)

		loss+=loss_i
		acc+=acc_i
	
	return loss/N, acc/N



def train(model=None, C=10, dim=784, lr=.01, batch_size=4096, num_epochs=15):
	""" instantiate and train an SVM classifier """

	# instantiate model
	if model is None:
		model = SVM(
			C,
			dim,
			lr=lr,
			decay=.995
		)

	# get dataset
	_, train_data, test_data = get_dataloaders(batch_size=batch_size)

	history={
		'train_loss':[],
		'train_acc':[],
		'test_loss':[],
		'test_acc':[],
		'w':[],
	}

	bar=tqdm(range(num_epochs*len(train_data)))
	
	# outer training loop is for each epoch
	for epoch in range(num_epochs):
		
		# inner loop iterates through the whole dataset
		for batch in train_data:
			# update progress bar
			bar.update()

			# split into data and labels. 
			data, labels = (
				batch['image'].detach().clone().numpy(), 
				batch['label'].detach().clone().numpy()
			)
				
			# compute loss
			loss, grad, acc = model.loss(data, labels)

			# collect historical data
			history['train_loss'].append(loss)
			history['train_acc'].append(acc)

			# optimizer step
			model.optim_step(grad)

			# collect historical data
			test_loss, test_acc = evaluate(model, test_data)
			history['test_loss'].append(test_loss)
			history['test_acc'].append(test_acc)
			history['w'].append(model.w.copy())
			
			# update progress bar
			bar.set_postfix({
				'loss_trn':history['train_loss'][-1],
				'loss_tst':history['test_loss'][-1],
				'acc_trn':history['train_acc'][-1],
				'acc_tst':history['test_acc'][-1],
				'lr':model.lr_decayed,
				'||w||':np.linalg.norm(history['w'][-1], ord=2),
			})
			


	return model, history
	
	return model
if __name__ == '__main__':
	train()
