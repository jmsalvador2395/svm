# The SVM Classifier.

This is an implementation of the SVM classifier on the MNIST dataset
\
\
Use [this](https://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf) resource for a brief introduction  
Use [this](https://www.robots.ox.ac.uk/~az/lectures/ml/matlab2.pdf) resource for implementation details in matlab


# Setup Instructions
This implementation is developed in python 3.9.15
Setup your python environment by following the instructions

#### With virtualenv
1. Change to the project directory. (the next commands require you to be here)
1. run `virtualenv env --python=<path to python 3.9 executable>`
1. run `source ./env/bin/activate`
1. run the bash script `pip_installs` or follow the last 2 steps. (they do the same thing)
1. run `pip install -r requirements.txt`
1. run `pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`


#### With Anaconda
1. change to the project directory. (the env.yml file is in the project root)
1. run `conda env create -f env.yml

# Usage Instructions
Using the environment built from the instructions above, start your jupyter notebook server by calling  
jupyter notebook` in the project root.  

Then you can follow the jupyter notebook to train a model, generate graphs from training, and then visualize the weights of the model
