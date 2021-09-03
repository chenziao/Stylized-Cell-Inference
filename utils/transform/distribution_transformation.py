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
