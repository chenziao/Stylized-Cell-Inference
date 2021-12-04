import numpy as np
import torch
from scipy.interpolate import griddata
from typing import Tuple, Optional, Union

# Project Imports
import cell_inference.config.params as params
from cell_inference.utils.spike_window import first_pk_tr


def build_lfp_grid(lfp: np.ndarray,
                   coord: np.ndarray,
                   grid_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a grid to match the Neuropixel Probe used to collect
    LFP signals: https://www.neuropixels.org/probe
    
    Then interpolate the LFP onto the grid for each timestep
    """
    t = lfp.shape[0]
    xy = coord[:, :2]
    xx, yy = np.meshgrid(grid_v[0], grid_v[1], indexing='ij')
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    grid_lfp = np.empty((t, grid.shape[0]))
    for i in range(t):
        grid_lfp[i, :] = griddata(xy, lfp[i, :], grid)
    return grid_lfp, grid


def calculate_stats(g_lfp: np.ndarray,
                    grid: np.ndarray = None) -> np.ndarray:
    """
    Calculates summary statistics. This includes:
        - Average Voltage of Each Channel
        - Standard Deviation Voltage of Each Channel
        - troughs of Each Channel
        - peaks of Each Channel
        - Difference between Peak and Trough amplitudes
        - Width of Waveform from half height (if grid specified)
    """
    g_lfp = np.asarray(g_lfp)
    grid_shape = tuple(v.size for v in params.ELECTRODE_GRID[:2])
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
            single_lfp_all_stats = np.array([mean, std, single_lfp_max_idx_x, sing_lfp_max_val])  # My,max_val])
        return single_lfp_all_stats

    def searchheights(lfp: np.ndarray, height: Optional[Union[float, int, np.ndarray]],
                      idx: Optional[Union[int, np.ndarray]]) -> Tuple[int, int]:
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

    def lfp_as_fy(lfp: np.ndarray, time: Optional[Union[int, np.ndarray]],
                  height: Optional[Union[float, int, np.ndarray]] = None) -> Tuple[int, int]:
        lfp_wrt_time = (lfp[time, :].reshape(4, 190))  # just removing the extra dimension for time with reshape
        x0_idx_wrt_time = np.argmax(np.max(np.abs(lfp_wrt_time), axis=1), axis=0)
        fy_wrt_x0_wrt_time = lfp_wrt_time[x0_idx_wrt_time, :]
        y0_idx_wrt_x0_wrt_time = np.argmax(np.abs(fy_wrt_x0_wrt_time), axis=0)
        half_height_fy_wrt_x0_wrt_time = np.abs(fy_wrt_x0_wrt_time[y0_idx_wrt_x0_wrt_time]) / 2 if height is None \
            else np.abs(fy_wrt_x0_wrt_time[y0_idx_wrt_x0_wrt_time])
        return searchheights(np.abs(fy_wrt_x0_wrt_time), half_height_fy_wrt_x0_wrt_time, y0_idx_wrt_x0_wrt_time)

    # Calculation of statistics across channels
    sl = [statscalc(x, i < i_min) for i, x in enumerate(stats_list)]
    #     sl = []

    """
    Calculates width of the first peak and adds it to the stats
    if grid parameter is provided
    """
    if grid is not None:
        reshaped_full_lfp = np.zeros((g_lfp.shape[0], 4, 190))
        for t in range(g_lfp.shape[0]):
            reshaped_full_lfp[t, :, :] = g_lfp[t, :].reshape(4, 190)
        ft_x = np.argmax(np.max(np.abs(reshaped_full_lfp), axis=2), axis=1)
        ft_y = np.argmax(np.max(np.abs(reshaped_full_lfp), axis=1), axis=1)
        ft_lfp = np.zeros((g_lfp.shape[0],))
        for i in range(g_lfp.shape[0]):
            ft_lfp[i] = reshaped_full_lfp[i, ft_x[i], ft_y[i]]

        t0 = first_pk_tr(g_lfp)
        t1 = t0
        for i in range(t0, g_lfp.shape[0] - t0):
            if ft_lfp[i] >= 0.0:  # TODO Fix this for the condition when it is an initial peak!!!
                t1 = i
                #                 print(y0)
                break

        t2 = first_pk_tr(g_lfp[t1:, :]) + t1

        idx_list = []
        idx_list.extend(lfp_as_fy(g_lfp, t0))
        idx_list.extend(lfp_as_fy(g_lfp, t1, height=0))
        idx_list.extend(lfp_as_fy(g_lfp, t2))
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
