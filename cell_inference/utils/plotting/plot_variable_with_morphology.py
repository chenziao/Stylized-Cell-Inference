import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows

def plot_variable_with_morphology(seg_coords, seg_prop, variable, t=None, axes = ['x', 'y'],
                                  distance_type='distance', n_dist=10, distance_range=None,
                                  select_seg=None, max_per_dist=None, varname='Variable',
                                  space=3, normalized_space=True, sort_by_dist=False,
                                  figsize=(15,12), fontsize=15, colormap='viridis', scalebar_size=50):

    nseg = seg_coords['r'].size

    # Axes for plotting morphology
    axes = np.asarray(axes)
    if  np.asarray(axes).dtype.type is np.str_:
        xyz = {'x': 0, 'y': 1, 'z': 2}
        axes = [xyz[axes[0]], xyz[axes[1]]]

    # Segment distance
    dist = seg_prop[distance_type] # endpoints
    dist05 = np.mean(dist, axis=1) # center

    # Segment radius
    soma_idx = seg_prop['swc_type']==1 # soma segment indices
    r = seg_coords['r'] / np.std(seg_coords['r'][~soma_idx]) # normalize radius
    r[soma_idx] = 2 * np.amax(r[~soma_idx])

    # Segment coordinates
    pc = seg_coords['pc'][:,axes] # center
    p01 = np.expand_dims(pc, 2) + np.expand_dims(seg_coords['dl'][:,axes], 2) * (np.array([[[-1,1]]]) / 2) # endpoints

    # Calculate distances at which variable traces are displayed
    if distance_range is not None:
        dmin, dmax = np.sort(distance_range)
    else:
        dmin, dmax = np.amin(dist), np.amax(dist)
    if dmin * dmax < 0:
        n_pos = int(np.floor((n_dist - 1) * dmax / (dmax - dmin))) + 1
        n_neg = n_dist - n_pos
        if distance_range is None:
            dmin = dmin - dmin / (2 * n_neg + 1)
            dmax = dmax - dmax / (2 * n_pos - 1)
        d_pts = np.concatenate((np.linspace(dmin, 0., n_neg, False), np.linspace(0., dmax, n_pos))) # include distance 0
    else:
        if distance_range is None:
            dmin, dmax = np.array([dmin, dmax]) + (dmax - dmin) / n_dist / 2 * np.array([1, -1])
        d_pts = np.linspace(dmin, dmax, n_dist)

    # Find segments at corresponding distances
    seg_idx = np.full(nseg, False)
    if select_seg is not None:
        ss = np.full(nseg, False)
        ss[select_seg] = True # selected segment index in boolean array
    dist[dist[:,1]==0,1] += 1e-6 # include segment with distance 0 on the right endpoint
    for p in d_pts:
        idx =  (dist[:,0]<=p) & (dist[:,1]>p)
        if select_seg is not None:
            idx &= ss # consider only selected segments
        if max_per_dist is not None:
            idx = np.nonzero(idx)[0][:max_per_dist] # consider only the first few segments
        seg_idx[idx] = True
    seg_idx = np.nonzero(seg_idx)[0].tolist()
    nshow = len(seg_idx) # number of segment to display


    # Color map
    sm = plt.cm.ScalarMappable(cmap=getattr(cm, colormap), norm=plt.Normalize())
    sm.set_array(dist05)
    sm.autoscale()

    # Figure
    fig = plt.figure(figsize=figsize)
    axs = []

    # Morphology
    ax = plt.subplot(1,3,1)
    axs.append(ax)
    # Plot
    for i in np.nonzero(~soma_idx)[0]:
        ax.plot(*p01[i], color=sm.to_rgba(dist05[i]), linewidth=r[i])
    for i in np.nonzero(soma_idx)[0]:
        ax.plot(*pc[i], color='r', marker='s', markersize=r[i])
    # Scale bar
    bar = AnchoredDirectionArrows(ax.transData, str(scalebar_size),  r'{} $\mu m$'.format(scalebar_size),
                                  length=scalebar_size, fontsize=fontsize, loc=6, color='k',
                                  sep_x=5, sep_y=5, back_length=0, head_width=0, head_length=0.01)
    ax.add_artist(bar)
    ax.axis('off')
    # Color bar
    cbar = fig.colorbar(sm, location='left', fraction=0.03, aspect=40, pad=0.)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(distance_type, size=fontsize)
    # Arrows head positions
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ypos = np.linspace(*yl, nshow, False)
    ypos += (yl[1] - yl[0]) / nshow / 2
    # Arrange order of segemnts at which variable traces are displayed
    if sort_by_dist:
        sorted_seg_idx = np.array(seg_idx)[np.argsort(dist05[seg_idx])]
    else:
        sorted_seg_idx = [seg_idx.pop(np.argmax((y - pc[seg_idx, 1]) / (xl[1] - pc[seg_idx, 0]))) for y in ypos]
    arrow = np.stack([pc[sorted_seg_idx], np.column_stack([np.full(nshow, xl[1]), ypos])], axis=2)
    # Arrows
    for arr in arrow:
        ax.plot(*arr[:,0], color='darkred', marker='o', ms=10, mfc='none', mew=1)
        ax.plot(*arr, color='grey', linewidth=1)
    ax.set_xlim(xl)
    ax.set_ylim(yl)

    # Variable traces
    ax = plt.subplot(1,3,(2,3))
    axs.append(ax)
    # Variable traces and time points
    X = variable[sorted_seg_idx,:]
    if t is None:
        t = np.arange(X.shape[1]) # time indices
    else:
        t = np.asarray(t)
        if t.size==1:
            t = t * np.arange(X.shape[1]) # interpret t as time step
    # Space between traces
    if normalized_space:
        Xmax = np.amax(np.abs(X), axis=1) # maximum magnitude in each trace
        Xh = space * np.std(Xmax[Xmax < space * np.std(Xmax)]) # remove outlier
    else:
        Xh = space
    xpos = Xh * np.arange(nshow) + Xh / 2 # height of each trace
    # Plot
    for i in range(nshow):
        d = dist05[sorted_seg_idx[i]]
        clr = sm.to_rgba(d)
        plt.plot(t[[0, -1]], np.full(2, xpos[i]), color='grey', linewidth=1) # 0 magnitude line
        plt.plot(t, xpos[i] + X[i], color=clr)
        plt.text(t[0], xpos[i], '%.3g' % (d), color=clr, fontsize=fontsize, verticalalignment='bottom')
    ax.set_xlim(t[[0, -1]])
    ax.set_ylim([0, nshow * Xh])
    ax.set_xlabel('Time (ms)', fontsize=fontsize)
    ax.set_ylabel(varname, fontsize=fontsize, labelpad=fontsize, rotation=270)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)

    fig.tight_layout(w_pad=-.5)
    plt.show()
    return fig, axs, sorted_seg_idx
