import numpy as np
from math import sqrt
from typing import Optional, Union, List, Tuple

def corrcoef(x: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> np.ndarray:
    """
    function for calculating correlation coefficient
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x -= np.mean(x)
    y -= np.mean(y)
    return np.sum(x * y) / sqrt(np.sum(x * x) * np.sum(y * y))

def max_corrcoef(x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                 window_size: Optional[int] = None) -> Tuple:
    """
    Calculate correlation coefficient between input x and y inside sliding time window (time should be the first
    axis, i.e. a column is a channel) Find the maximum correlation coefficient and the corresponding window location.
    Return first the maximum value, then return the index of the start point of corresponding window. If window_size
    is not specified, use the length of x as window size. Return the index of window in y. If window_size is
    specified, slide the window separately in x and y. Return the indices of windows in x and y respectively.
    """
    if window_size is None:
        win = x.shape[0]
    else:
        win = window_size
    nx = x.shape[0] - win + 1
    ny = y.shape[0] - win + 1
    corr = np.empty((nx, ny))
    for i in range(nx):
        for j in range(ny):
            corr[i, j] = corrcoef(x[i:i + win, :].ravel(), y[j:j + win, :].ravel())
    maxind = np.argmax(corr)
    maxind = np.unravel_index(maxind, (nx, ny))
    max_corr = corr[maxind]
    if window_size is None:
        output = (max_corr, maxind[1], None)
    else:
        output = (max_corr, maxind[0], maxind[1])
    return output
