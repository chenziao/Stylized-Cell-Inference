import numpy as np
from scipy.stats import norm

def norm2unif(x,a,b):
    """Transform normal distributed variables to follow uniform [a,b] distribution"""
    y = a+(b-a)*norm.cdf(np.asarray(x))
    return y

def range2logn(a,b,n_sigma=2):
    """Get the lognormal parameters given range [a,b] in linear scale that corresponds to n_sigma"""
    A = np.log(a)
    B = np.log(b)
    mu = (A+B)/2
    sigma = (B-A)/n_sigma/2
    return mu, sigma

def norm2logn(x,mu,sigma):
	"""Transform normal distributed variables to follow lognormal (mu,sigma) distribution"""
	y = np.exp(mu+sigma*x)
	return y


# 'logds' stands for log probability density offset by variable transformation
# E.g. if variable x has probability density f(x) and variable y=g(x),
# then y has probatility density f(x)*|dx/dy|, and log density log(f(x))+log(|dx/dy|)
# logds calculates log(|dx/dy|).

def logds_norm2unif(x,a,b):
	logds = -np.log(b-a)-np.log(norm.pdf(x))
	return logds

def logds_norm2logn(x,mu,sigma):
	logds = -np.log(sigma)-mu-sigma*x
	return logds
