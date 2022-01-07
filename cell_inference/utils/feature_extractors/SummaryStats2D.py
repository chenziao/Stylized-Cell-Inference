import numpy as np
import torch
from scipy.interpolate import griddata
from typing import Tuple, Optional, Union

# Project Imports
import cell_inference.config.params as params

GRID_SHAPE = tuple(v.size for v in params.ELECTRODE_GRID[:2])


def build_lfp_grid(lfp: np.ndarray,
                   coord: np.ndarray,
                   grid_v: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                   y_window_size: Optional[Union[float, int, np.ndarray]] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a grid to match the Neuropixel Probe used to collect
    LFP signals: https://www.neuropixels.org/probe
    
    Then interpolate the LFP onto the grid for each timestep
        y_window_size - If specified, get the grid within a window along y-axis instead,
        with the window centered at the maximum amplitude location. (micron)
        Return None if the window falls outside the given electrode array range.
    """
    t = lfp.shape[0]
    xy = coord[:, :2]
    grid_y = grid_v[1]
    if y_window_size is not None:
        # relative index of window in y grid
        dy = abs(grid_y[-1] - grid_y[0]) / (GRID_SHAPE[1] - 1)
        ny = max(int(np.floor(y_window_size/dy)) + 1, 3)
        rel_idx = int(np.floor((ny - 1) / 2)) + np.arange(ny, dtype=int)
        # find maximum amplitude location
        max_idx = np.argmax(np.amax(np.abs(lfp), axis=0))
        center_y = xy[max_idx, 1]
        center_y_idx = np.argmin(np.abs(grid_y - center_y))
        if center_y_idx + rel_idx[0] < 0 or center_y_idx + rel_idx[-1] > GRID_SHAPE[1]:
            raise ValueError("The window falls outside the given electrode array range.")
        else:
            grid_y = grid_y[center_y_idx + rel_idx]
    xx, yy = np.meshgrid(grid_v[0], grid_y, indexing='ij')
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    grid_lfp = np.empty((t, grid.shape[0]))
    for i in range(t):
        grid_lfp[i, :] = griddata(xy, lfp[i, :], grid)
    return grid_lfp, grid


def calculate_stats(g_lfp: np.ndarray,
                    grid: np.ndarray = None,
                    additional_stats: bool = True) -> np.ndarray:
    """
    Calculates summary statistics. This includes:
        - Average Voltage of Each Channel
        - Standard Deviation Voltage of Each Channel
        - troughs of Each Channel
        - peaks of Each Channel
        - Difference between Peak and Trough time
        - Width of Waveform from half height (if additional_stats True)
    """
    g_lfp = np.asarray(g_lfp)
    grid_shape = (GRID_SHAPE[0], int(g_lfp.shape[1]/GRID_SHAPE[0]))
    avg = np.mean(g_lfp, axis=0)  # average voltage of each channel
    std_dev = np.std(g_lfp, axis=0)  # stDev of the voltage of each channel
    t_t = np.argmin(g_lfp, axis=0)
    t_p = np.argmax(g_lfp, axis=0)

    troughs = -np.take_along_axis(g_lfp, np.expand_dims(t_t, axis=0), axis=0)
    peaks = np.take_along_axis(g_lfp, np.expand_dims(t_p, axis=0), axis=0)
    rel_t = t_p - t_t

    stats_list = [avg, rel_t, std_dev, troughs, peaks]
    i_min = 2  # include minimum statistics for the first i_min in stats_list

    """
    Helper functions for calculating statistics across channels and searching for
    a specified height
        - statscalc: calculates the statistics across each channel
            - include_min: include minimun value and position
            
        - searchheights: calculates width of waveform given a height
    """

    def statscalc(stats: np.ndarray, include_min: bool = True) -> np.ndarray:
        stats = stats.ravel()
        mean = np.mean(stats)
        std = np.std(stats)
        single_lfp_max_idx = np.argmax(stats)
        sing_lfp_max_val = stats[single_lfp_max_idx]
        single_lfp_max_idx_x, single_lfp_max_idx_y = np.unravel_index(single_lfp_max_idx, grid_shape)
        if include_min:
            single_lfp_min_idx = np.argmin(stats)
            single_lfp_min_val = stats[single_lfp_min_idx]
            single_lfp_min_idx_x, single_lfp_min_idx_y = np.unravel_index(single_lfp_min_idx, grid_shape)
            single_lfp_all_stats = np.array([mean, std, single_lfp_max_idx_x,
                                             single_lfp_max_idx_y, sing_lfp_max_val,
                                             single_lfp_min_idx_x, single_lfp_min_idx_y, single_lfp_min_val])
        else:
            single_lfp_all_stats = np.array([mean, std, single_lfp_max_idx_x,
                                             single_lfp_max_idx_y, sing_lfp_max_val])  # My,max_val])
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

    def half_height_width_wrt_y(lfp: np.ndarray) -> Tuple[int, int]:
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

    # Calculation of statistics across channels
    sl = [statscalc(x, i < i_min) for i, x in enumerate(stats_list)]
    #     sl = []

    """
    Calculates width of the first peak and adds it to the stats
    if grid parameter is provided
    """
    if additional_stats:
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
        idx_list.extend(half_height_width_wrt_y(g_lfp[t0, :]))
        idx_list.extend(half_height_width_wrt_y(g_lfp[t2, :]))

        t1_stats = statscalc(g_lfp[t1, :], include_min=True)
        idx_list.extend((t1_stats[3], t1_stats[6]))
        idx_list.extend((t0, t1, t2))
        sl += [np.array(idx_list)]

    all_stats = np.concatenate(sl)
    return all_stats


def cat_output(lfp: np.ndarray,
               include_sumstats=True) -> torch.Tensor:
    """
    Driver function. Creates grid and interpolates provided LFP then concatenates
    calculated summary statistics to the interpolated LFP

    Always include summary stats unless specified not to
    """
    g_lfp, grid = build_lfp_grid(lfp, params.ELECTRODE_POSITION, params.ELECTRODE_GRID)
    output = np.concatenate((g_lfp.ravel(), calculate_stats(g_lfp, grid))) if include_sumstats else lfp.ravel()
    return torch.from_numpy(output)
