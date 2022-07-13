import numpy as np
from typing import Tuple


def first_pk_tr(lfp: np.ndarray) -> np.ndarray:
    """
    Find the time index of first peak/trough in "lfp" (2D array, each column is a channel).
    """
    return min(first_tr(lfp), first_pk(lfp))


def first_pk(lfp: np.ndarray) -> np.ndarray:
    """
    Find the time index of first peak in "lfp" (2D array, each column is a channel).
    """
    max_idx = np.argmax(np.amax(lfp, axis=1))
    return max_idx


def first_tr(lfp: np.ndarray) -> np.ndarray:
    """
    Find the time index of first trough in "lfp" (2D array, each column is a channel).
    """
    min_idx = np.argmin(np.amin(lfp, axis=1))
    return min_idx


def get_spike_window(lfp: np.ndarray,
                     win_size: int,
                     align_at: int = 0) -> Tuple[int, int]:
    """
    Get the window of the spike waveform, where the first peak/trough is aligned at a fixed point in the window.
    lfp: input lfp with spike waveform (2D array, each column is a channel)
    win_size: window size (time samples)
    align_at: time index in the window to align with the first peak/trough in "lfp"
    return (start,end), the time index of the window in "lfp"
    """
    lfp_time = lfp.shape[0]
    if win_size > lfp_time:
        raise ValueError("win_size cannot be greater than length of lfp")
    align_pt = first_pk_tr(lfp)
    start = max(align_pt - align_at, 0)
    end = start + win_size
    if end > lfp_time:
        raise ValueError("End of the window exceeds the data time frame")
    return start, end
