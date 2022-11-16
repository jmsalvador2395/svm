import numpy as np

class SVM:
	""" implementation of the SVM classifier """

	def __init__(self, C, dim, loc=0, scale=1):
		"""
		initialize the weights

		:param C:		number of classes
		:param dim:		dimensionality of the input
		:param loc:		arguement for initialization using numpy.random.normal()
		:param scale:	standard deviation for initialization using numpy.random.normal()
		"""

		# save dimensionality for reshaping input
		self.dim=dim

		# initialize weights. use +1 for the bias weights
		self.w=np.random.normal(loc=loc, scale=scale, size=(dim+1, C))
	
	def __call__(self, x):
		"""
		generates classifier scores

		:param x:	the input data
		"""

		# get number of samples
		N=x.shape[0]

		# reshape x to an Nxdim matrix and then pad with 1s
		xpad=np.hstack((
			x.reshape(N, self.dim),
			np.ones(N)[:, None]
		))

		return xpad@self.w

if __name__ == '__main__':
	""" main function for testing class implementation """

	N, C, dim = (20, 5, 4)
	svm=SVM(C, dim)
	x=np.random.randn(N, dim)

	# should output 20x5
	print(svm(x).shape)

	# check shape of output
	assert svm(x).shape == (N, C), "incorrect output shape"
