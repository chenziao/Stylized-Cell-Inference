import numpy as np

def log_modulus(x, x0=1.):
    """
    Log modulus transfomr
    x0: reference magnitude (x0 > 0), values beyond which get spreaded out. x0 |-> 1, x0 maps to unit value.
    """
    if x0<=0:
        raise ValueError("Parameter x0 must be positive.")
    return np.sign(x) * (np.log2(np.abs(x) + x0) - np.log2(x0))

def log_moulus_nfold(x, n_fold=31.):
    """Apply transform with reference value being 1/n_fold of the maximum magnitude"""
    return log_modulus(x, np.amax(np.abs(x)) / n_fold)
