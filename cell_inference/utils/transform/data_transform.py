import numpy as np


def log_modulus(x, x0=1):
    """
    Log modulus transfomr
    x0: reference magnitude (x0 > 0), values beyond which get spreaded out. x0 |-> 1, x0 maps to unit value.
    """
    if x0<=0:
        raise ValueError("Parameter x0 must be positive.")
    return np.sign(x) * (np.log2(np.abs(x) + x0) - np.log2(x0))
