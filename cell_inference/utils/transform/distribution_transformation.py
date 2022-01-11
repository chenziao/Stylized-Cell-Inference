import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple, Union


def norm2unif(x: Optional[Union[float, int, np.ndarray]],
              a: Optional[Union[float, int, np.ndarray]],
              b: Optional[Union[float, int, np.ndarray]]) -> np.ndarray:
    """Transform normal distributed variables to follow uniform [a,b] distribution"""
    y = a + (b - a) * norm.cdf(np.asarray(x))
    return y

def range2norm(a: Optional[Union[float, int, np.ndarray]],
               b: Optional[Union[float, int, np.ndarray]],
               n_sigma: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Get the normal parameters given range [a,b] that corresponds to n_sigma"""
    mu = (a + b) / 2
    sigma = np.abs(b - a) / n_sigma / 2
    return mu, sigma

def range2logn(a: Optional[Union[float, int, np.ndarray]],
               b: Optional[Union[float, int, np.ndarray]],
               n_sigma: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Get the lognormal parameters given range [a,b] in linear scale that corresponds to n_sigma"""
    return range2norm(np.log(a), np.log(b), n_sigma)


def norm2logn(x: Optional[Union[float, int, np.ndarray]],
              mu: Optional[Union[float, int, np.ndarray]],
              sigma: Optional[Union[float, int, np.ndarray]]) -> np.ndarray:
    """Transform normal distributed variables to follow lognormal (mu,sigma) distribution"""
    y = np.exp(mu + sigma * x)
    return y


# 'logds' stands for log probability density offset by variable transformation
# E.g. if variable x has probability density f(x) and variable y=g(x),
# then y has probatility density f(x)*|dx/dy|, and log density log(f(x))+log(|dx/dy|)
# logds calculates log(|dx/dy|).

def logds_norm2unif(x: Optional[Union[float, int, np.ndarray]],
                    a: Optional[Union[float, int, np.ndarray]],
                    b: Optional[Union[float, int, np.ndarray]]) -> np.ndarray:
    logds = -np.log(b - a) - np.log(norm.pdf(x))
    return logds


def logds_norm2logn(x: Optional[Union[float, int, np.ndarray]],
                    mu: Optional[Union[float, int, np.ndarray]],
                    sigma: Optional[Union[float, int, np.ndarray]]) -> np.ndarray:
    logds = -np.log(sigma) - mu - sigma * x
    return logds
