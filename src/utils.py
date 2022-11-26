from pathlib import Path
import argparse

def get_project_root():
	return str(Path(__file__).parent.parent)

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--dataset',
		type = str,
		default = 'pg19',
		help = 'the huggingface dataset to use. (default: pg19)'
	)

	parser.add_argument(
		'--dataset_dir',
		type = str,
		default = 'datasets',
		help = (
			'the directory for downloading/reading the huggingface dataset. ' + 
			'the input directory is relative to the root of the project'
		)
	)

	parser.add_argument(
		'--batch_size',
		type = int,
		default = 16,
		help = 'the batch size for training on the dataset'
	)

	parser.add_argument(
		'--num_epochs',
		type = int,
		default = 5,
		help = 'the number of epochs for training on the dataset'
	)

	
	args = parser.parse_args()
	return vars(args)
