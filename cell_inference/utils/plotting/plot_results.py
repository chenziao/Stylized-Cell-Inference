import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from typing import Union, Optional, Tuple, List
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter


def plot_lfp_traces(t: np.ndarray, lfp: np.ndarray, savefig: Optional[str] = None,
                    fontsize: int = 40, labelpad: int = -30,
                    tick_length: int = 15, nbins: int = 3) -> Tuple[Figure, Axes]:
    """
    Plot LFP traces.

    Parameters
    t: time points (ms). 1D array
    lfp: LFP traces (uV). If is 2D array, each column is a channel.
    savefig: if specified as string, save figure with the string as file name.
    fontsize: size of font for display
    labelpad: Spacing in points from the Axes bounding box including ticks and tick labels.
    tick_length: length between ticks
    nbins: number of bins to create
    """
    t = np.asarray(t)
    lfp = np.asarray(lfp)
    fig = plt.figure()  # figsize=(15,15))
    if lfp.ndim == 2:
        legend_elements = []
        for j in range(lfp.shape[1]):
            line, = plt.plot(t, lfp[:, j])
            legend_elements.append(line)
    else:
        line = plt.plot(t, lfp)
    plt.xlabel('ms', fontsize=fontsize)
    plt.ylabel('LFP (\u03bcV)', fontsize=fontsize, labelpad=labelpad)
    plt.locator_params(axis='both', nbins=nbins)
    plt.tick_params(length=tick_length, labelsize=fontsize)
    ax = plt.gca()
    plt.show()
    if savefig is not None:
        if type(savefig) is not str:
            savefig = 'LFP_trace.pdf'
        fig.savefig(savefig, bbox_inches='tight', transparent=True)
    return fig, ax


def plot_lfp_heatmap(t: np.ndarray, elec_d: np.ndarray, lfp: np.ndarray, savefig: Optional[str] = None,
                     vlim: str = 'auto', fontsize: int = 40, ticksize: int = 30, labelpad: int = -12, nbins: int = 3,
                     cbbox: Optional[List[float]] = None, cmap: str = 'viridis') -> Tuple[Figure, Axes]:
    """
    Plot LFP heatmap.

    t: time points (ms). 1D array
    elec_d: electrode distance (um). 1D array
    lfp: LFP traces (uV). If is 2D array, each column is a channel.
    savefig: if specified as string, save figure with the string as file name.
    vlim: value limit for color map, using +/- 3-sigma of lfp for bounds as default. Use 'max' for maximum bound range.
    fontsize: size of font for display
    labelpad: Spacing in points from the Axes bounding box including ticks and tick labels.
    tick_length: length between ticks
    nbins: number of bins to create
    cbbox: dimensions of figure
    cmap: A Colormap instance or registered colormap name. The colormap maps the C values to color.
    """
    if cbbox is None:
        cbbox = [.91, 0.118, .03, 0.76]
    lfp = np.asarray(lfp).T
    elec_d = np.asarray(elec_d) / 1000
    if type(vlim) is str:
        if vlim == 'max':
            vlim = [np.min(lfp), np.max(lfp)]
        else:
            vlim = 3 * np.std(lfp) * np.array([-1, 1])
    fig, ax = plt.subplots()
    pcm = plt.pcolormesh(t, elec_d, lfp, cmap=cmap, vmin=vlim[0], vmax=vlim[1], shading='auto')
    cbaxes = fig.add_axes(cbbox)
    cbar = fig.colorbar(pcm, ax=ax, ticks=np.linspace(vlim[0], vlim[1], nbins), cax=cbaxes)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('LFP (\u03bcV)', fontsize=fontsize, labelpad=labelpad)
    ax.set_xticks(np.linspace(t[0], t[-1], nbins))
    ax.set_yticks(np.linspace(elec_d[0], elec_d[-1], nbins))
    ax.tick_params(labelsize=ticksize)
    ax.set_xlabel('time (ms)', fontsize=fontsize)
    ax.set_ylabel('dist_y (mm)', fontsize=fontsize)
    plt.show()
    if savefig is not None:
        if type(savefig) is not str:
            savefig = 'LFP_heatmap.pdf'
        fig.savefig(savefig, bbox_inches='tight', transparent=True)
    return fig, ax


def plot_multiple_lfp_heatmaps(t: np.ndarray, elec_d: np.ndarray, lfp: np.ndarray, savefig: Optional[str] = None,
                               vlim: Union[List, str] = 'auto', fontsize: int = 40, ticksize: int = 30,
                               labelpad: int = -12, nbins: int = 3, cmap: str = 'viridis',
                               fig: Optional[Figure] = None, outer: GridSpec = None, col: int = 0,
                               cell_num: int = 0, title: str = '') -> Union[str, List[float]]:
    """
    Plot LFP heatmap.

    t: time points (ms). 1D array
    elec_d: electrode distance (um). 1D array
    lfp: LFP traces (uV). If is 2D array, each column is a channel.
    savefig: if specified as string, save figure with the string as file name.
    vlim: value limit for color map, using +/- 3-sigma of lfp for bounds as default. Use 'max' for maximum bound range.
    fontsize: size of font for display
    labelpad: Spacing in points from the Axes bounding box including ticks and tick labels.
    tick_length: length between ticks
    nbins: number of bins to create
    cbbox: dimensions of figure
    cmap: A Colormap instance or registered colormap name. The colormap maps the C values to color.
    """
    lfp = np.asarray(lfp).T * 7720
    elec_d = np.asarray(elec_d) / 1000
    if type(vlim) is str:
        if vlim == 'max':
            vlim = [np.min(lfp), np.max(lfp)]
        else:
            vlim = 3 * np.std(lfp) * np.array([-1, 1])

    gs = GridSpecFromSubplotSpec(1, 15, subplot_spec=outer[cell_num, col], wspace=0.25)
    cbaxes = fig.add_subplot(gs[14])
    ax = fig.add_subplot(gs[:14])
    lfp = np.asarray(lfp)
    # elec_d = np.asarray(elec_d) / 1000
    pcm = ax.pcolormesh(t, elec_d, lfp, cmap=cmap, vmin=vlim[0], vmax=vlim[1], shading='auto')
    cbar = fig.colorbar(pcm, ax=ax, ticks=np.linspace(vlim[0], vlim[1], nbins), cax=cbaxes, format='%.2f')
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('LFP (\u03bcV)', fontsize=fontsize, labelpad=labelpad)
    # print(t[-1])
    ax.set_xticks(np.linspace(t[0], t[-1], nbins))
    ax.set_yticks(np.linspace(elec_d[0], elec_d[-1], nbins))
    ax.tick_params(labelsize=ticksize)
    ax.set_xlabel('time (ms)', fontsize=fontsize)
    ax.set_ylabel('dist_y (mm)', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return vlim


# TODO Needs to have the cell membrane voltage be a 2D array instead of just the soma
def plot_intracellular_spike_heatmap(t: np.ndarray, elec_d: np.ndarray, intracellular_spikes: np.ndarray,
                                     savefig: Optional[str] = None, vlim: Union[str, Tuple, List, np.ndarray] = 'auto',
                                     fontsize: int = 40, ticksize: int = 30, labelpad: int = -12, nbins: int = 3,
                                     cbbox: Optional[List[float]] = None, cmap: str = 'viridis') -> Tuple[Figure, Axes]:
    """
    Plot Intracellular Spike heatmap.

    t: time points (ms). 1D array
    elec_d: electrode distance (um). 1D array
    intracellular_spikes: intracellular_spikes
    savefig: if specified as string, save figure with the string as file name.
    vlim: value limit for color map, using +/- 3-sigma of lfp for bounds as default. Use 'max' for maximum bound range.
    fontsize: size of font for display
    labelpad: Spacing in points from the Axes bounding box including ticks and tick labels.
    tick_length: length between ticks
    nbins: number of bins to create
    cbbox: dimensions of figure
    cmap: A Colormap instance or registered colormap name. The colormap maps the C values to color.
    """
    if cbbox is None:
        cbbox = [.91, 0.118, .03, 0.76]
    intracellular_spikes = np.asarray(intracellular_spikes).T
    elec_d = np.asarray(elec_d) / 1000
    if type(vlim) is str:
        if vlim == 'max':
            vlim = [np.min(intracellular_spikes), np.max(intracellular_spikes)]
        else:
            vlim = 3 * np.std(intracellular_spikes) * np.array([-1, 1])
    fig, ax = plt.subplots()
    pcm = plt.pcolormesh(t, elec_d, intracellular_spikes, cmap=cmap, vmin=vlim[0], vmax=vlim[1], shading='auto')
    cbaxes = fig.add_axes(cbbox)
    cbar = fig.colorbar(pcm, ax=ax, ticks=np.linspace(vlim[0], vlim[1], nbins), cax=cbaxes)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('Intracellular Spikes (\u03bcV)', fontsize=fontsize, labelpad=labelpad)
    ax.set_xticks(np.linspace(t[0], t[-1], nbins))
    ax.set_yticks(np.linspace(elec_d[0], elec_d[-1], nbins))
    ax.tick_params(labelsize=ticksize)
    ax.set_xlabel('time (ms)', fontsize=fontsize)
    ax.set_ylabel('dist_y (mm)', fontsize=fontsize)
    plt.show()
    if savefig is not None:
        if type(savefig) is not str:
            savefig = 'intracellular_spikes_heatmap.pdf'
        fig.savefig(savefig, bbox_inches='tight', transparent=True)
    return fig, ax
