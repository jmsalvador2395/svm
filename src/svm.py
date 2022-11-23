import numpy as np
from numpy.linalg import norm

class SVM:
	""" implementation of the SVM classifier """

	def __init__(self, C, dim, shape=(28, 28), reg=1, lr=1, loc=0, scale=1):
		"""
		initialize the weights

		:param C: number of classes
		:param dim: dimensionality of the input
		:param reg: the regularization term
		:param loc: arguement for initialization using numpy.random.normal()
		:param scale: standard deviation for initialization using numpy.random.normal()
		"""


		# save the learning rate
		self.lr=lr

		# save the regularization term
		self.reg=reg

		# save dimensionality for reshaping input
		self.dim=dim

		# save the shape of the data
		self.shape=shape

		# initialize weights. use +1 for the bias weights
		self.w=np.random.normal(loc=loc, scale=scale, size=(dim+1, C))

		# set bias weights to 0
		self.w[-1, :]=0
	
	def __call__(self, x):
		"""
		generates classifier scores. computes x@w+b

		:param x: the input data

		:return: prediction scores
		"""

		# get number of samples
		N=x.shape[0]

		# reshape x to an Nxdim matrix and then pad with 1s
		xpad=np.hstack((
			x.reshape(N, self.dim),
			np.ones(N)[:, None]
		))

		return xpad@self.w

	def predict(x):
		""" 
		returns the label predictions

		:param x: the input data

		:return: a numpy array of the predicted labels. size (N, )
		"""
		scores=self(x)
		return np.argmax(scores, axis=-1)

	def loss(self, x, y=None):
		"""
		this is used for calculating the loss function
		
		:param x: the input data
		:param y: the labels for the input data

		:return loss: the output of the loss function
		:return grad: the weight gradient
		"""

		# get the number of samples
		N=x.shape[0]

		# reshape x to an Nxdim matrix and then pad with 1s
		xpad=np.hstack((
			x.reshape(N, self.dim),
			np.ones(N)[:, None]
		))

		# use this to index the scores
		xi=range(N)

		# calculate scores
		scores=self(x)

		preds=np.argmax(scores, axis=-1)

		# compute accuracy
		acc=np.mean(preds==y)
		#acc=np.sum(preds==y)

		# compute loss
		ys=-np.ones(scores.shape)
		ys[xi, y]=1
		svm_scores=np.maximum(0, 1-ys*scores)
		loss=np.mean(svm_scores)+self.reg*norm(self.w, ord=2)

		# compute gradient
		mask=(svm_scores==0)
		grad=-ys
		grad[mask]=0
		grad=xpad.T@grad/grad.size+self.reg*2*self.w

		return loss, grad, acc
	
	def optim_step(self, grad):
		"""
		use this for the optimization step. Implementation is SGD

		:param grad: the gradient of the current training step
		"""
		self.w-=self.lr*grad

	def w_norm(ord=2):
		"""return the sum of norms of the weight vectors"""
		return np.linalg.norm(self.w, ord=2)
	

if __name__ == '__main__':
	""" main function for testing class implementation """

	N, C, dim = (20, 5, 4)
	svm=SVM(C, dim)
	x=np.random.randn(N, dim)

	# should output 20x5
	print(svm(x).shape)

	# check shape of output
	assert svm(x).shape == (N, C), "incorrect output shape"
