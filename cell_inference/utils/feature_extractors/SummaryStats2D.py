import numpy as np
import torch
from scipy.spatial import qhull
from scipy import signal
from sklearn.linear_model import HuberRegressor
from typing import Tuple, List, Optional, Union

# Project Imports
import cell_inference.config.params as params
from cell_inference.utils.spike_window import get_spike_window

GRID = params.ELECTRODE_GRID
GRID_SHAPE = tuple(v.size for v in GRID)

# Process 2D LFP array
def get_y_window(lfp: np.ndarray, coord: np.ndarray,
                 y_window_size: float = params.Y_WINDOW_SIZE,
                 grid_v: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get a window along y-axis centered at the maximum amplitude location.
        lfp: LFP array with shape (time x channels)
        coord: Coordinates of each channel (channels x 2)
        y_window_size: The width of the window along y-axis (micron)
        grid_v: The grid vectors we use to get the window index. (x grid, y grid, z grid)
            If not specified, use params.ELECTRODE_GRID.
    Return Windowed grid y coordinates, Index of the window in the grid, y coordinates of window center
    """
    grid_y = grid_v[1]
    y_size = grid_y.size
    # relative index of window in y grid
    dy = abs(grid_y[-1] - grid_y[0]) / (y_size - 1)
    ny = max(int(np.floor(y_window_size / 2 / dy) * 2) + 1, 3)
    rel_idx = np.arange(-(ny - 1) / 2,(ny + 1) / 2, dtype=int)
    # find maximum amplitude location
    max_idx = np.argmax(np.amax(np.abs(lfp), axis=0))
    center_y = coord[max_idx, 1]
    center_y_idx = np.argmin(np.abs(grid_y - center_y))
    if center_y_idx + rel_idx[0] < 0 or center_y_idx + rel_idx[-1] >= y_size:
        raise ValueError("The window falls outside the given electrode grid range.")
    y_window_idx = center_y_idx + rel_idx
    grid_y = grid_y[y_window_idx]
    return grid_y, y_window_idx, center_y

def build_lfp_grid(lfp: np.ndarray, coord: np.ndarray,
                   y_window_size: Optional[float] = None,
                   grid_v: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
                   ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build a grid to match the Neuropixel Probe used to collect
    LFP signals: https://www.neuropixels.org/probe
    
    Then interpolate the LFP onto the grid for each timestep
        y_window_size - If specified, get the grid within a window along y-axis instead,
            with the window centered at the maximum amplitude location. (micron)
            Raise error if the window falls outside the given electrode array range.
            Return y coordinates of window center in the last output if specified, otherwise None
    Return Gridded LFP array with shape (time x channels), Grid coordinates (channels x 2), y window center
    """
    if grid_v is None:
        grid_v = GRID
    if y_window_size is None:
        grid_y = grid_v[1]
        center_y = None
    else:
        grid_y, _ , center_y = get_y_window(lfp=lfp, coord=coord, y_window_size=y_window_size, grid_v=grid_v)
    xx, yy = np.meshgrid(grid_v[0], grid_y, indexing='ij')
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    t = lfp.shape[0]
    grid_lfp = np.empty((t, grid.shape[0]))
    vertices, weights = interp_weights(coord[:, :2], grid)
    for i in range(t):
        grid_lfp[i, :] = interpolate(lfp[i, :], vertices, weights)
    return grid_lfp, grid, center_y

def interp_weights(points, xi) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the indices of the vertices of the enclosing simplex and the weights for the interpolation"""
    points = np.asarray(points)
    d = points.shape[1]
    tri = qhull.Delaunay(points)
    simplex = tri.find_simplex(xi)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = xi - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    return vertices, weights

def interpolate(values, vertices, weights, fill_value=None) -> np.ndarray:
    """Calculate interpolation values"""
    vi = np.einsum('nj,nj->n', np.take(values, vertices), weights)
    if fill_value is not None:
        vi[np.any(weights < 0, axis=1)] = fill_value
    return vi


def get_lfp_y_window(g_lfp: np.ndarray, coord: np.ndarray,
                     y_window_size: float = params.Y_WINDOW_SIZE,
                     grid_v: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get LFP within a window along y-axis centered at the maximum amplitude location.
        g_lfp: Gridded LFP array with shape (time x channels)
        grid_v: The grid vectors we use to get the window index. (x grid, y grid, z grid)
            If not specified, use params.ELECTRODE_GRID.
            Raise error if grid is not consistent with g_lfp channel dimensions.
    Return LFP array within y window with shape (time x channels), y coordinates of window center
    """
    if grid_v is None:
        grid_v = GRID
        grid_shape = GRID_SHAPE
    else:
        grid_shape = tuple(v.size for v in grid_v)
    if g_lfp.shape[1] != grid_shape[0]*grid_shape[1]:
        raise ValueError("LFP array is not consistent with grid shape.")
    _, y_window_idx, center_y = get_y_window(lfp=g_lfp, coord=coord, y_window_size=y_window_size, grid_v=grid_v)
    t = g_lfp.shape[0]
    lfp_y_window = g_lfp.reshape((t, grid_shape[0], grid_shape[1]))[:,:,y_window_idx].reshape((t, -1))
    return lfp_y_window, center_y


# Summary statistics
def calculate_stats(g_lfp: np.ndarray, additional_stats: int = 1,
                    grid_shape: Optional[Tuple[int]] = None) -> np.ndarray:
    """
    Calculates summary statistics. This includes:
        - Average Voltage of Each Channel
        - Trough Time of Each Channel
        - Peak Time of Each Channel
        - Standard Deviation Voltage of Each Channel
        - Troughs of Each Channel
        - Peaks of Each Channel
        - Width of Waveform from half height (if additional_stats True)
        g_lfp: gridded LFP array with shape (time x channels)
        grid_shape: grid shape of g_lfp.
            If not specified, use x size in params.ELECTRODE_GRID and infer y size automatically.
    """
    g_lfp = np.asarray(g_lfp) * 1000  # convert from mV to uV
    if grid_shape is None:
        grid_shape = (GRID_SHAPE[0], int(g_lfp.shape[1]/GRID_SHAPE[0]))
    else:
        grid_shape = grid_shape[:2]
    if g_lfp.shape[1] != grid_shape[0]*grid_shape[1]:
        raise ValueError("LFP array is not consistent with grid shape.")

    avg = np.mean(g_lfp, axis=0)  # average voltage of each channel
    std_dev = np.std(g_lfp, axis=0)  # stDev of the voltage of each channel
    t_t, t_p, troughs, peaks = get_tr_pk(g_lfp)

    stats_list = [avg, t_t, t_p, std_dev, troughs, peaks]
    i_min = 3  # include minimum statistics for the first i_min in stats_list

    # Calculation of statistics across channels
    ss = [statscalc(x, grid_shape, i < i_min) for i, x in enumerate(stats_list)]

    if additional_stats >= 1:
        """Calculate time and the widths of the major trough and peak"""
        # Trough and Peak times with maximum amplitude
        t_T = t_t[np.argmax(troughs)]
        t_P = t_p[np.argmax(peaks)]
        t0 = min(t_T, t_P)
        t2 = max(t_T, t_P)

        # Find channel with maximum amplitude
        max_idx = np.argmax(np.amax(np.abs(g_lfp), axis=0))
        # Find time when LFP changes sign
        t_idx = np.nonzero(np.diff(np.sign(g_lfp[t0:t2, max_idx])))[0]
        t1 = t0 + 1 + t_idx[0] if t_idx.size > 0 else t0

        idx_list = []
        idx_list.extend((t0, t1, t2))
        idx_list.extend(half_height_width_wrt_y(g_lfp[t0, :], grid_shape))
        idx_list.extend(half_height_width_wrt_y(g_lfp[t2, :], grid_shape))
        t1_stats = statscalc(g_lfp[t1, :], grid_shape, include_min=True)
        idx_list.extend((t1_stats[3], t1_stats[6]))
        ss += [np.array(idx_list)]

    if additional_stats >= 2:
        """Calculate the decay of trough and peak along y direction"""
        lambda_troughs = get_decay(get_max_val_y(troughs, grid_shape))
        lambda_peaks = get_decay(get_max_val_y(peaks, grid_shape))
        ss += [np.array(lambda_troughs + lambda_peaks)]

    return np.concatenate(ss)

def get_tr_pk(g_lfp: np.ndarray) -> Tuple:
    t_t = np.argmin(g_lfp, axis=0)  # trough time
    t_p = np.argmax(g_lfp, axis=0)  # peak time
    troughs = -np.take_along_axis(g_lfp, np.expand_dims(t_t, axis=0), axis=0)  # trough magnitude
    peaks = np.take_along_axis(g_lfp, np.expand_dims(t_p, axis=0), axis=0)  # peak magnitude
    return t_t, t_p, troughs, peaks

def statscalc(stats: np.ndarray, grid_shape: Tuple[int], include_min: bool = True) -> np.ndarray:
    """
    Helper functions for calculating statistics across channels and searching for
    a specified height
        - statscalc: calculates the statistics across each channel
            - include_min: include minimun value and position
            
        - searchheights: calculates width of waveform given a height
    """
    stats = stats.ravel()
    mean = np.mean(stats)
    std = np.std(stats)
    single_lfp_max_idx = np.argmax(stats)
    single_lfp_max_val = stats[single_lfp_max_idx]
    single_lfp_max_idx_x, single_lfp_max_idx_y = np.unravel_index(single_lfp_max_idx, grid_shape)
    if include_min:
        single_lfp_min_idx = np.argmin(stats)
        single_lfp_min_val = stats[single_lfp_min_idx]
        single_lfp_min_idx_x, single_lfp_min_idx_y = np.unravel_index(single_lfp_min_idx, grid_shape)
        single_lfp_all_stats = np.array([mean, std, single_lfp_max_idx_x,
                                         single_lfp_max_idx_y, single_lfp_max_val,
                                         single_lfp_min_idx_x, single_lfp_min_idx_y, single_lfp_min_val])
    else:
        single_lfp_all_stats = np.array([mean, std, single_lfp_max_idx_x,
                                         single_lfp_max_idx_y, single_lfp_max_val])  # My,max_val])
    return single_lfp_all_stats

def searchheights(lfp: np.ndarray, height: Union[float, int, np.ndarray], idx: int) -> Tuple[int, int]:
    idx_left, idx_right = 0, lfp.size
    for i in range(idx - 1, idx_left, -1):
        if lfp[i] <= height:
            idx_left = i
            break
    for i in range(idx + 1, idx_right):
        if lfp[i] <= height:
            idx_right = i
            break
    return idx_left, idx_right

def half_height_width_wrt_y(lfp: np.ndarray, grid_shape: Tuple[int]) -> Tuple[int, int]:
    # channel with max amplitude
    idx_wrt_time = np.argmax(np.abs(lfp))
    # max amplitude
    height = lfp[idx_wrt_time]
    # half height of max amplitude
    half_height = np.abs(height)/2
    # x, y index of max channel
    x0_idx_wrt_time, y0_idx_wrt_time = np.unravel_index(idx_wrt_time, grid_shape)
    # make sure height in fy is positive
    fy_wrt_x0_wrt_time = np.sign(height) * lfp.reshape(grid_shape)[x0_idx_wrt_time, :]
    return searchheights(fy_wrt_x0_wrt_time, half_height, y0_idx_wrt_time)

def get_max_val_y(x: np.ndarray, grid_shape: Tuple[int]) -> np.ndarray:
    """Get maximum values along x for each y"""
    x = x.reshape(grid_shape)  # variable for each channel in 2D array
    y = np.take_along_axis(x, np.expand_dims(np.argmax(x, axis=0), axis=0), axis=0).ravel()
    return y

HbReg = HuberRegressor(epsilon=4., alpha=0)  # robust against outlier (epsilon in unit of natural log of magnitude)
def get_decay(y: np.ndarray) -> List[float]:
    """
    Length constant of decay along -/+ y from maximum value (unit: grid spacing along y)
    measured by linear regression and slope, respectively.
    Lambda - by fitting log of magnitude vs. grid index
    Slope - by slope between the maximum value point and the point on the edge
    """
    max_idx = np.argmax(y)
    Lambda = []
    Slope = []
    # calculate for left and right sides respectively
    for y_one_side in (y[:max_idx + 1], y[max_idx:]):
        indices = np.nonzero(y_one_side > 0)[0]
        log_y = np.log(y_one_side[indices])
        try:
            c = HbReg.fit(indices.reshape(-1, 1), log_y).coef_[0]
        except: # in case there is convergence error
            c, _ = np.polyfit(indices, log_y, deg=1)
        Lambda.append(1 / abs(c))
        Slope.append( abs((indices[-1] - indices[0]) / (log_y[-1] - log_y[0])) ) # slope y-range/log(ratio)
    return Lambda + Slope


def scaled_stats_indices(boolean: bool = False, additional_stats: int = 1) -> np.ndarray:
    """
    Return indices of summary statistics that scales with LFP magnitude.
    Return a boolean list if boolean is True.
    Edit this function if needed when summary statistics change.
    """
    # mean, std, single_lfp_max_val, (single_lfp_min_val)
    n_stats = lambda include_min: 8 if include_min else 5  # number of stats across channels
    stats_idx = lambda include_min: [0, 1, 4, 7] if include_min else [0, 1, 4]
    # stats_list = [avg, t_t, t_p, std_dev, troughs, peaks]
    i_min = 3  # include minimum statistics for the first i_min in stats_list
    scale_stats = np.array([1, 0, 0, 1, 1, 1], dtype=bool)  # whether stats (for each channel) in the list scales
    n = 0
    indices = []
    for i, scale in enumerate(scale_stats):
        if scale:
            for j in stats_idx(i < i_min):
                indices.append(n + j)
        n += n_stats(i < i_min)
    if additional_stats >= 1:
        n += 9
    if additional_stats >= 2:
        n += 8
    indices = np.array(indices)
    if boolean:
        indices_bool = np.full(n, False)
        indices_bool[indices] = True
        indices = indices_bool
    return indices


def cat_output(lfp: np.ndarray, include_sumstats=True) -> torch.Tensor:
    """
    Driver function. Creates grid and interpolates provided LFP then concatenates
    calculated summary statistics to the interpolated LFP.
    Always include summary stats unless specified not to.
    """
    g_lfp, grid = build_lfp_grid(lfp, params.ELECTRODE_POSITION, params.ELECTRODE_GRID)
    output = np.concatenate((g_lfp.ravel(), calculate_stats(g_lfp, grid))) if include_sumstats else lfp.ravel()
    return torch.from_numpy(output)


# All processes together
filt_b, filt_a = signal.butter(params.BUTTERWORTH_ORDER,
                               params.FILTER_CRITICAL_FREQUENCY,
                               params.BANDFILTER_TYPE,
                               fs=params.FILTER_SAMPLING_RATE)

def process_lfp(lfp: np.ndarray, coord: np.ndarray = params.ELECTRODE_POSITION, dt: float = params.DT,
                pad_spike_window: bool = False, align_at: int = params.PK_TR_IDX_IN_WINDOW,
                y_window_size: float = params.Y_WINDOW_SIZE, ycoord: float = None,
                calc_summ_stats: bool = True, additional_stats: int = 1, err_msg: bool = False) -> Tuple:
    """
    Process LFP: filter, find spike window, interpolate in grid, window along y-axis, shift in y, summary statistics.
    
    lfp: LFP array with shape (time x channels)
    coord: Coordinates of each channel (channels x 2 or 3)
    dt: time step for generating time array. Do not generate if set to None
    pad_spike_window: whether or not to pad when LFP duration is too short for time window
    align_at: keyword argument "align_at" for the function "get_spike_window"
    y_window_size: The width of the window along y-axis (micron)
    ycoord: the y-coordinate of the soma. Specify to calculate the y-shift
    calc_summ_stats: whether or not to calculate summary statistics
    err_msg: whether or not to print error message
    
    Return: bad case id, LFP array, time array, electrode coordinates,
            y coordinates of window center, y-shift, summary statistics
    """
    bad = -2
    while bad < 0:
        try:
            filtered_lfp = signal.lfilter(filt_b, filt_a, lfp, axis=0) # filter along time axis
            # filtered_lfp /= np.max(np.abs(filtered_lfp))
            start, end = get_spike_window(filtered_lfp, win_size=params.WINDOW_SIZE, align_at=align_at)
        except ValueError as e:
            if err_msg: print(e)
            if pad_spike_window:
                bad = -1
                pad_size = (params.PK_TR_IDX_IN_WINDOW, params.WINDOW_SIZE - params.PK_TR_IDX_IN_WINDOW)
                lfp = np.pad(lfp, (pad_size, (0, 0)), 'linear_ramp', end_values=((0,),))
            else:
                bad = 1
                t = None if dt is None else dt * np.arange(filtered_lfp.shape[0])
        else:
            windowed_lfp = filtered_lfp[start:end,:]
            t = None if dt is None else dt * np.arange(params.WINDOW_SIZE)
            if bad == -1:
                break
            else:
                bad = 0

    if bad != 1:
        try:
            g_lfp, g_coords, y_c = build_lfp_grid(windowed_lfp, coord, y_window_size=y_window_size)
        except ValueError as e:
            if err_msg: print(e)
            bad = 2

    if bad==1:
        output = (bad, filtered_lfp, t, coord, None, None, None)
    elif bad==2:
        output = (bad, windowed_lfp, t, coord, None, None, None)
    else:
        yshift = None if ycoord is None or y_c is None else y_c - ycoord
        summ_stats = calculate_stats(g_lfp, additional_stats=additional_stats) if calc_summ_stats else None
        output = (bad, g_lfp, t, g_coords, y_c, yshift, summ_stats)
    return output
