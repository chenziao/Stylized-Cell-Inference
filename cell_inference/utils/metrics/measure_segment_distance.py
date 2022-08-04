import numpy as np
from neuron import h

def measure_segment_distance(soma, section_list, sec_type, freq=0, extracellular_mechanism=True):
    seg_prop = {}
    swc_type = []
    seg_area = []
    seg_dist = []
    seg_length = []
    elec_dist = []
    elec_dist0 = [1]

    # set up distance origin
    h.distance(0,soma(.5))
    # set up electrotonic origin
    zz = h.Impedance()
    zz.loc(soma(.5))
    if extracellular_mechanism:
        zz.compute(freq)
    else:
        zz.compute(freq, 1)

    # measure distance
    for i, sec in enumerate(section_list):
        if sec_type[i]!=1:
            elec_dist0.append(zz.ratio(sec.parentseg()))
        for j, seg in enumerate(sec):
            swc_type.append(sec_type[i])
            seg_area.append(seg.area())
            seg_dist.append(h.distance(seg))
            seg_length.append(sec.L/sec.nseg)
            if j!=0:
                elec_dist0.append(elec_dist[-1])
            elec_dist.append(zz.ratio(seg))
    seg_prop['swc_type'] = np.array(swc_type)
    seg_prop['seg_area'] = np.array(seg_area)

    # distance at endpoints of each segment
    seg_prop['distance'] = np.expand_dims(seg_dist, 1) + np.expand_dims(seg_length, 1) / 2 * np.array([[-1, 1]])
    seg_prop['elec_dist'] = np.sort(-np.log(np.column_stack([elec_dist0, elec_dist])), axis=1)

    # change sign of basal and axon types
    idx = np.nonzero((seg_prop['swc_type']==2) | (seg_prop['swc_type']==3))[0]
    seg_prop['distance'][idx] = -seg_prop['distance'][idx,::-1]
    seg_prop['elec_dist'][idx] = -seg_prop['elec_dist'][idx,::-1]
    return seg_prop, zz.transfer(soma(.5)), zz.transfer_phase(soma(.5))
