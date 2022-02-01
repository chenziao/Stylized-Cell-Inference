import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Union, Optional, List, Tuple

from cell_inference.utils.transform.geometry_transformation import pol2cart, hphi2unitsphere

def plot_all_cells(df: pd.DataFrame, s: float = 10, w: float = 2,
                   axes: Union[List[int], Tuple[int]] = [2, 0, 1],
                   view: Union[str, List[float], Tuple[float, float]] = '3D',
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
    if not set(['y','h','phi','r_s','l_t','r_t']).issubset(set(df.columns)):
        raise ValueError("Lacking features of position or morphology.")
    
    if 'x' not in df.columns or 'z' not in df.columns:
        if 'd' not in df.columns or 'theta' not in df.columns:
            raise ValueError("Lacking features of position or morphology.")
        else:
            df['x'], df['z'] = pol2cart(df['d'], df['theta'])
    
    p0 = df[['x', 'y', 'z']].values
    p1 = p0 + hphi2unitsphere(df[['h', 'phi']].values) * df[['l_t']].values
    
    r_s = s / df['r_s'].mean() * df['r_s'].values
    r_t = w / df['r_t'].mean() * df['r_t'].values
    
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize())
    sm.set_array(r_t)
    sm.autoscale()
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    
    somas = ax.scatter(*p0[:, axes].T, c=r_s, s=r_s, cmap='plasma')
    for i in range(len(df)):
        trunks = ax.plot3D(*np.vstack((p0[i], p1[i]))[:,axes].T, color=sm.to_rgba(r_t[i]), linewidth=r_t[i])
    fig.colorbar(sm, fraction=0.1, shrink=0.5, aspect=30, label='trunk radius (um)')
    fig.colorbar(somas, fraction=0.1, shrink=0.5, aspect=30, label='soma radius (um)')
    
    xyz = 'xyz'
    ax.set_xlabel(xyz[axes[0]] + ' (um)')
    ax.set_ylabel(xyz[axes[1]] + ' (um)')
    ax.set_zlabel(xyz[axes[2]] + ' (um)')
    
    box = np.vstack((np.vstack((p0, p1)).min(axis=0), np.vstack((p0, p1)).max(axis=0)))
    center = box.mean(axis=0)
    box = center + (box[1] - box[0]).max() / 2 * np.array([[-1], [1]])
    ax.auto_scale_xyz(*box.T[axes])
    
    if type(view) is str:
        if view == '2D':
            ax.view_init(elev=0, azim=0)
    else:
        ax.view_init(elev=view[0], azim=view[1])
    
    plt.show()
    return fig, ax