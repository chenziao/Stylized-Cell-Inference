import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Union, Optional, List, Tuple

from cell_inference.utils.transform.geometry_transformation import pol2cart, hphi2unitsphere


def plot_all_cells(df: pd.DataFrame, s: float = 10, w: float = 2,
                   axes=None, view: Union[str, List[float], Tuple[float, float]] = '3D',
                   figsize: Optional[Tuple[float, float]] = None) -> Tuple[Figure, Axes]:
    """
    Plot multiple cells location, orientation and simplified morphology in 3D,
    using ball and stick to represent soma and trunk.
    
    df: Pandas dataframe that contains parameters of all cells.
    s: average marker size for soma.
    w: average line width for trunk.
    axes: sequence of axes to display in 3d plot axes.
        Default: [2,0,1] show z,x,y in 3d plot x,y,z axes, so y is upward.
    view: Camera position. Default: '3D' automatic 3D view. '2D' show xy. Or tuple of (elev, azim)
    Return Figure object, Axes object
    """
    if axes is None:
        axes = [2, 0, 1]
    df = df.copy()
    if not {'y', 'h', 'phi'}.issubset(set(df.columns)):
        raise ValueError("Lacking features of position or morphology.")

    if 'x' not in df.columns or 'z' not in df.columns:
        if 'd' not in df.columns or 'theta' not in df.columns:
            raise ValueError("Lacking features of position or morphology.")
        else:
            df['x'], df['z'] = pol2cart(df['d'], df['theta'])
    if 'r_s' in df.columns:
        r_s = s / df['r_s'].mean() * df['r_s'].values
    else:
        r_s = np.full(len(df), s)
    if 'r_t' in df.columns:
        r_t = w / df['r_t'].mean() * df['r_t'].values
    else:
        r_t = np.full(len(df), w)
    if 'l_t' in df.columns:
        l_t = df[['l_t']].values
    else:
        l_t = np.full((len(df), 1), 300.)

    p0 = df[['x', 'y', 'z']].values
    p1 = p0 + hphi2unitsphere(df[['h', 'phi']].values) * l_t

    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize())
    sm.set_array(r_t)
    sm.autoscale()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    somas = ax.scatter(*p0.T[axes], c=r_s, s=r_s, cmap='plasma')
    for i in range(len(df)):
        trunks = ax.plot3D(*np.vstack((p0[i], p1[i])).T[axes], color=sm.to_rgba(r_t[i]), linewidth=r_t[i])
    if 'r_t' in df.columns:
        fig.colorbar(sm, fraction=0.1, shrink=0.5, aspect=30, label='trunk radius')
    if 'r_s' in df.columns:
        fig.colorbar(somas, fraction=0.1, shrink=0.5, aspect=30, label='soma radius')

    xyz = 'xyz'
    ax.set_xlabel(xyz[axes[0]] + ' (um)')
    ax.set_ylabel(xyz[axes[1]] + ' (um)')
    ax.set_zlabel(xyz[axes[2]] + ' (um)')

    box = np.vstack((np.vstack((p0, p1)).min(axis=0), np.vstack((p0, p1)).max(axis=0)))
    center = box.mean(axis=0)
    box = center + (box[1] - box[0]).max() / 2 * np.array([[-1], [1]])
    ax.set_xlim(box.T[axes[0]])
    ax.set_ylim(box.T[axes[1]])
    ax.set_zlim(box.T[axes[2]])

    if type(view) is str:
        if view == '2D':
            ax.view_init(elev=0, azim=-90)
    else:
        ax.view_init(elev=view[0], azim=view[1])

    return fig, ax
