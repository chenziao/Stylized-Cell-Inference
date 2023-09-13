import numpy as np
import torch
from scipy import signal
from scipy.spatial import qhull
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from typing import Tuple, List, Optional, Union

# Project Imports
import cell_inference.config.params as params
from cell_inference.utils.spike_window import get_spike_window
from cell_inference.utils.transform.data_transform import log_modulus

GRID = params.ELECTRODE_GRID
GRID_SHAPE = tuple(v.size for v in GRID)
DY = abs(GRID[1][-1] - GRID[1][0]) / (GRID[1].size - 1)
NY = max(int(np.floor(params.Y_WINDOW_SIZE / 2 / DY) * 2) + 1, 5)
REL_IDX = np.arange(-(NY - 1) / 2,(NY + 1) / 2, dtype=int)

# Process 2D LFP array
def get_y_window(lfp: np.ndarray, coord: np.ndarray, y_window_size: float = params.Y_WINDOW_SIZE,
                 grid_v: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]] = GRID
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
    if y_window_size is params.Y_WINDOW_SIZE and grid_v is GRID:
        rel_idx = REL_IDX
    else:
        dy = DY if grid_v is GRID else abs(grid_y[-1] - grid_y[0]) / (y_size - 1)
        ny = max(int(np.floor(y_window_size / 2 / dy) * 2) + 1, 5)
        rel_idx = np.arange(-(ny - 1) / 2,(ny + 1) / 2, dtype=int)
    # find maximum amplitude location
    max_idx = np.argmax(np.amax(np.abs(lfp), axis=0))
    center_y = coord[max_idx, 1]
    center_y_idx = np.argmin(np.abs(grid_y - center_y))
    y_window_idx = center_y_idx + rel_idx
    if y_window_idx[0] < 0 or y_window_idx[-1] >= y_size:
        raise ValueError("The window (%d, %d) falls outside the given electrode grid range."
                         % tuple(y_window_idx[[0, -1]]))
    return grid_y[y_window_idx], y_window_idx, center_y

def build_lfp_grid(lfp: np.ndarray, coord: np.ndarray, y_window_size: Optional[float] = None,
                   grid_v: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = GRID
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
    if g_lfp.shape[1] != grid_shape[0] * grid_shape[1]:
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
        - Width of Waveform from half height (if additional_stats >= 1)
        - Other statistics if additional_stats > 1
        g_lfp: gridded LFP array with shape (time x channels) (mV)
        grid_shape: grid shape of g_lfp.
            If not specified, use x size in params.ELECTRODE_GRID and infer y size automatically.
    """
    g_lfp = np.asarray(g_lfp) * 1000  # convert from mV to uV
    if grid_shape is None:
        grid_shape = (GRID_SHAPE[0], int(g_lfp.shape[1] / GRID_SHAPE[0]))
    else:
        grid_shape = grid_shape[:2]
    if g_lfp.shape[1] != grid_shape[0] * grid_shape[1]:
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
        # Find channel with maximum amplitude
        max_idx = np.argmax(np.fmax(troughs, peaks))

        # Trough and Peak times with maximum amplitude
        t_T = t_t[max_idx]
        t_P = t_p[max_idx]
        t0 = min(t_T, t_P)
        t2 = max(t_T, t_P)

        # Find time when LFP changes sign
        t_idx = np.nonzero(np.diff(np.sign(g_lfp[t0:t2, max_idx])))[0]
        t1 = t0 + 1 + t_idx[0] if t_idx.size > 0 else t0

        idx_list = []
        idx_list.extend((t0, t1, t2))
        idx_list.extend(half_height_width_wrt_y(g_lfp[t0, :], grid_shape))
        idx_list.extend(half_height_width_wrt_y(g_lfp[t2, :], grid_shape))
        t1_stats = statscalc(g_lfp[t1, :], grid_shape, include_min=True)
        idx_list.extend((t1_stats[3], t1_stats[6]))
        ss.append(idx_list)

    if additional_stats >= 2:
        """Calculate the decay of trough and peak along y direction"""
        lambda_troughs, pts_troughs, tr_max_idx = get_decay(get_max_val_y(troughs, grid_shape))
        lambda_peaks, pts_peaks, pk_max_idx = get_decay(get_max_val_y(peaks, grid_shape))
        ss.append(lambda_troughs + lambda_peaks)

    if additional_stats >= 3:
        """Add decay break point and average magnitude around it"""
        max_mag = max(ss[4][4], ss[5][4])  # Global maximum
        tr_avg_mag, _ = volume_average(g_lfp, t_t, troughs, pts_troughs, tr_max_idx, grid_shape, max_mag=max_mag)
        pk_avg_mag, _ = volume_average(g_lfp, t_p, peaks, pts_peaks, pk_max_idx, grid_shape, max_mag=max_mag)
        ss.extend(pts_troughs + pts_peaks)
        ss.append(tr_avg_mag + pk_avg_mag)

    if additional_stats >= 4:
        """Time of trough and peak propogation along y direction from the center to the break point"""
        t_t_m = get_max_val_t(t_t, troughs, grid_shape)
        t_p_m = get_max_val_t(t_p, peaks, grid_shape)
        prop_pts_tr, prop_time_tr, prop_y2_tr = get_prop_time(t_t_m, tr_max_idx, pts_troughs)
        prop_pts_pk, prop_time_pk, prop_y2_pk = get_prop_time(t_p_m, pk_max_idx, pts_peaks)
        ss.extend(prop_pts_tr + prop_pts_pk)
        ss.append(prop_time_tr + prop_time_pk)

    if additional_stats >= 5:
        """Statistics after logmodulus transform on LFP"""
        n_fold = 31
        ref_mag = max(troughs[max_idx], peaks[max_idx]) / n_fold
        lfp_logmod = log_modulus(g_lfp, ref_mag)
        log_avg = np.mean(lfp_logmod, axis=0)  # average voltage of each channel
        log_std_dev = np.std(lfp_logmod, axis=0)  # stDev of the voltage of each channel
        log_troughs = -np.take_along_axis(lfp_logmod, t_t[np.newaxis], axis=0).ravel()  # trough magnitude
        log_peaks = np.take_along_axis(lfp_logmod, t_p[np.newaxis], axis=0).ravel()  # peak magnitude
        log_stats_list = [log_avg, log_std_dev, log_troughs, log_peaks]
        ss.extend([statscalc(x, grid_shape, include_max=False) for x in log_stats_list])

    return np.concatenate(ss)

def statscalc(stats: np.ndarray, grid_shape: Tuple[int],
              include_min: bool = True, include_max: bool = True) -> np.ndarray:
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
    all_stats = [mean, std]
    if include_max:
        max_channel_idx = np.argmax(stats)
        max_channel_val = stats[max_channel_idx]
        max_channel_idx_x, max_channel_idx_y = np.unravel_index(max_channel_idx, grid_shape)
        all_stats.extend([max_channel_idx_x, max_channel_idx_y, max_channel_val])
        if include_min:
            min_channel_idx = np.argmin(stats)
            min_channel_val = stats[min_channel_idx]
            min_channel_idx_x, min_channel_idx_y = np.unravel_index(min_channel_idx, grid_shape)
            all_stats.extend((min_channel_idx_x, min_channel_idx_y, min_channel_val))
    return np.array(all_stats)

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

def get_tr_pk(g_lfp: np.ndarray) -> Tuple:
    t_t = np.argmin(g_lfp, axis=0)  # trough time
    t_p = np.argmax(g_lfp, axis=0)  # peak time
    troughs = -np.take_along_axis(g_lfp, t_t[np.newaxis], axis=0).ravel()  # trough magnitude
    peaks = np.take_along_axis(g_lfp, t_p[np.newaxis], axis=0).ravel()  # peak magnitude
    return t_t, t_p, troughs, peaks

def get_max_idx_x(m: np.ndarray) -> np.ndarray:
    """Get index of maximum values of input m along x for each y"""
    return np.argmax(m, axis=0)

def get_max_val_y(m: np.ndarray, grid_shape: Tuple[int]) -> np.ndarray:
    """Get maximum values of input m along x for each y"""
    m = m.reshape(grid_shape)  # variable for each channel in 2D array
    my = np.take_along_axis(m, get_max_idx_x(m)[np.newaxis], axis=0).ravel()
    return my

def get_max_val_t(t, m: np.ndarray, grid_shape: Tuple[int]) -> np.ndarray:
    """Get time of maximum values of input m along x for each y"""
    t = t.reshape(grid_shape)
    m = m.reshape(grid_shape)
    mt = np.take_along_axis(t, get_max_idx_x(m)[np.newaxis], axis=0).ravel()
    return mt

def line(y, y1, w1, y2, w2):
    """Two-point form of a line"""
    return w1 + (w2 - w1) / (y2 - y1) * (y - y1)

# First line segment (fixing one point at origin)
def line1(y, y1, w1):
    return line(y, 0., 0., y1, w1)

def two_line_segments(y, w1, w2, y1, y2=100.):
    # Second line segment (fixing y2)
    def line2(y, y1, w1):
        return line(y, y2, w2, y1, w1)
    return np.piecewise(y, [y <= y1], [line1, line2], y1, w1)

def get_decay(my, bound=7.0):
    """
    Fit two-segment lines to capture the decay of magnitude along -/+ y from maximum value.
    bound: minimum value clip for log relative magnitude (bound by exp(-bound) of maximum magnitude).
    Return length constant of decay (unit: grid spacing along y) measured by lambda and slope, respectively.
    Lambda - by inverse slope (grid number / log magnitude) of the first line segment. 
    Slope - by slope between the maximum value point and the point on the edge (y2 = half of y window size).
    Return the point coordinates representation of the fit two-segment lines (w1, w2, y1).
    """
    PTS = []
    Lambda = []
    Slope = []
    y2 = np.floor(my.size / 2)
    fn = lambda y, w1, w2, y1: two_line_segments(y, w1, w2, y1, y2=y2)
    max_idx = np.argmax(my)
    for my_one_side in (my[max_idx::-1], my[max_idx:]):
        bounds = ((.1, .1, 1.), (bound, bound, min(my_one_side.size - 1, y2) - 1.))
        log_my = np.log(my_one_side[0]) - np.log(np.fmax(my_one_side, my_one_side[0] * np.exp(-bound)))  # e^-7 < 1/1000
        pts, _ = curve_fit(fn, np.arange(my_one_side.size, dtype=float), log_my,
                           p0=np.mean(np.array(bounds), axis=0), bounds=bounds,
                           method='dogbox', loss='huber', f_scale=bound / 2)
        PTS.append(pts)
        Lambda.append(pts[2] / pts[0])
        Slope.append(y2 / pts[1])
    return Lambda + Slope, PTS, max_idx

def get_fit(my, PTS, max_idx=None):
    """Get fit lines and breakpoints index and magnitude (log scale)"""
    if max_idx is None:
        max_idx = np.argmax(my)
    log_max = np.log(my[max_idx])
    fn = lambda y, pts: log_max - two_line_segments(y, *pts, y2=np.floor(my.size / 2))
    w = (fn(np.arange(max_idx, 0, -1, dtype=float), PTS[0]), fn(np.arange(0, my.size - max_idx, dtype=float), PTS[1]))
    break_point = np.array([[max_idx], [log_max]]) - np.array([[PTS[0][2], -PTS[1][2]], [PTS[0][0], PTS[1][0]]])
    return np.concatenate(w), break_point

VOLUME_RANGE = ((-9, 31), (0, 1), (0, 15))  # t, x, y
VOLUME_RANGE = np.tile(np.array(VOLUME_RANGE).T, (2, 1, 1))
VOLUME_RANGE[0, :, 2] = 1 - VOLUME_RANGE[0, ::-1, 2]  # flip y for reversed direction
def volume_average(lfp, t_m, m, PTS, max_idx, grid_shape, max_mag=None):
    """
    Return average lfp magnitude in the neighbor volume near the break point relative to the global maximum.
    lfp: lfp array
    t_m, m: returned by 'get_tr_pk'
    PTS, max_idx: returned by 'get_decay'
    """
    shape_3d = (lfp.shape[0],) + grid_shape
    lfp = lfp.reshape((-1,) + grid_shape)
    m = m.reshape(grid_shape)
    t_m = t_m.reshape(grid_shape)
    if max_mag is None:
        max_mag = np.amax(np.abs(lfp))
    avg_mag = []
    brkpnt_idx = []
    for i in range(2):
        y1 = int(PTS[i][2])
        y = max_idx + (y1 if i else -y1)
        x = np.argmax(m[:, y])
        t = t_m[x, y]
        box_range = np.clip(np.array((t, x, y)) + VOLUME_RANGE[i], 0, shape_3d).T
        box = tuple(slice(idx[0], idx[1]) for idx in box_range)
        avg_mag.append(np.mean(lfp[box]) / max_mag)
        brkpnt_idx.append((t, x, y))
    return avg_mag, brkpnt_idx

def line2pt_func(y1, y2):
    def line2pt(y, w1, w2):
        return line(y, y1, w1, y2, w2)
    return line2pt

def get_prop_time(mt, max_idx, PTS, t_margin=0.2/params.DT):
    side = np.array([-1, 1])
    y1 = max_idx + side
    y2 = max_idx + side * np.array([int(pts[2]) for pts in PTS])
    TPTS = []
    prop_time = []
    for i in range(2):
        y_dist = (y2[i] - y1[i]) * side[i]
        if y_dist < 1:
            y2[i] = y1[i] + 1 * side[i]
        y_idx = np.arange(y1[i], y2[i] + side[i], side[i])
        fn = line2pt_func(max_idx, y2[i])
        pts, _ = curve_fit(fn, y_idx.astype(float), mt[y_idx],
                           p0=mt[y_idx[[0, -1]]], bounds=(0, float(params.WINDOW_SIZE)),
                           method='dogbox', loss='huber', f_scale=t_margin)
        TPTS.append(pts)
        prop_time.append((pts[1] - pts[0]) / y_dist)
    return TPTS, prop_time, y2

def get_prop_fit(max_idx, y2, TPTS):
    side = np.array([-1, 1])
    y_idx = []
    time_lines = []
    for i in range(2):
        y_idx.append(np.arange(max_idx, y2[i] + side[i], side[i])[::side[i]])
        time_lines.append(line(y_idx[i].astype(float), max_idx, TPTS[i][0], y2[i], TPTS[i][1]))
    return np.concatenate(time_lines), np.concatenate(y_idx)

def scaled_stats_indices(boolean: bool = False, additional_stats: int = 1) -> np.ndarray:
    """
    Return indices of summary statistics that scales with LFP magnitude.
    Return a boolean list if boolean is True.
    Edit this function if needed when summary statistics change.
    """
    # mean, std, max_channel_val, (min_channel_val)
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
    if additional_stats >= 3:
        n += 16
    if additional_stats >= 4:
        n += 12
    if additional_stats >= 5:
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
                filter: bool = True, y_window_size: float = params.Y_WINDOW_SIZE, ycoord: float = None,
                gauss_filt: bool = False, calc_summ_stats: bool = True, additional_stats: int = 1,
                err_msg: bool = False, err_full = False) -> Tuple:
    """
    Process LFP: filter, find spike window, interpolate in grid, window along y-axis, shift in y, summary statistics.
    
    lfp: LFP array with shape (time x channels) or (spikes x time x channels)
    coord: Coordinates of each channel (channels x 2 or 3)
    dt: time step for generating time array. Do not generate if set to None
    pad_spike_window: whether or not to pad when LFP duration is too short for time window
    align_at: keyword argument "align_at" for the function "get_spike_window"
    filter: whether or not to use temporal filter
    y_window_size: The width of the window along y-axis (micron)
    ycoord: the y-coordinate of the soma. Specify to calculate the y-shift
    gauss_filt: whether or not to use 3D Gaussian filter
    calc_summ_stats: whether or not to calculate summary statistics
    additional_stats: series number of additional summary statistics
    err_msg: whether or not to print error message
    
    Return: bad case id, LFP array, time array, electrode coordinates,
            y coordinates of window center, y-shift, summary statistics
    """
    lfp = np.asarray(lfp)
    if lfp.ndim == 2:
        bad, windowed_lfp = get_t_window(lfp, pad_spike_window, align_at, filter, err_msg)
    elif lfp.ndim == 3:
        nspk = lfp.shape[0]
        bad = np.empty(nspk)
        windowed_lfp = [None] * nspk
        for i in range(nspk):
            bad[i], windowed_lfp[i] = get_t_window(lfp[i], pad_spike_window, align_at, filter, err_msg)
        if err_full: bad_full = bad
        bad = 0 if np.all(bad == 0) else (-1 if pad_spike_window else 1)
        windowed_lfp = np.mean(lfp if bad == 1 else windowed_lfp, axis=0) 
    else:
        raise ValueError('LFP array must be a 2D or 3D array.')

    t = None if dt is None else dt * np.arange(windowed_lfp.shape[0])
    if bad < 1:
        try:
            g_lfp, g_coords, y_c = build_lfp_grid(windowed_lfp, coord, y_window_size=y_window_size)
        except ValueError as e:
            if err_msg: print(e)
            bad = 2
        else:
            if gauss_filt:
                sigma = (2, 1, .8)
                grid_shape = (g_lfp.shape[0], GRID_SHAPE[0], int(g_lfp.shape[1] / GRID_SHAPE[0]))
                g_lfp = gaussian_filter(g_lfp.reshape(grid_shape), sigma, truncate=3, mode='nearest').reshape((grid_shape[0], -1))

    if bad < 1:
        yshift = None if ycoord is None or y_c is None else y_c - ycoord
        summ_stats = None
        if calc_summ_stats:
            try:
                summ_stats = calculate_stats(g_lfp, additional_stats=additional_stats)
            except Exception as e:
                if err_msg: print(e)
                bad = 3
        output = (bad, g_lfp, t, g_coords, y_c, yshift, summ_stats)
    else:
        output = (bad, windowed_lfp, t, coord, None, None, None)
    if err_full:
        output += (bad_full,)
    return output

def get_t_window(lfp, pad_spike_window=False, align_at=params.PK_TR_IDX_IN_WINDOW, filter=True, err_msg=False):
    bad = -2
    while bad < 0:
        try:
            filtered_lfp = signal.lfilter(filt_b, filt_a, lfp, axis=0) if filter else lfp # filter along time axis
            start, end = get_spike_window(filtered_lfp, win_size=params.WINDOW_SIZE, align_at=align_at)
        except ValueError as e:
            if err_msg: print(e)
            if pad_spike_window:
                bad = -1
                pad_size = (params.PK_TR_IDX_IN_WINDOW, params.WINDOW_SIZE - params.PK_TR_IDX_IN_WINDOW)
                lfp = np.pad(lfp, (pad_size, (0, 0)), 'linear_ramp', end_values=((0,),))
            else:
                bad = 1
                windowed_lfp = filtered_lfp
        else:
            windowed_lfp = filtered_lfp[start:end, :]
            if bad == -1:
                break
            else:
                bad = 0
    return bad, windowed_lfp
