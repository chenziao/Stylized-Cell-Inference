from neuron import h
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional, List, Tuple, Union

from cell_inference.utils.currents.recorder import Recorder
from cell_inference.cells.stylizedcell import StylizedCell


class EcpMod(object):
    """
    A module for recording single cell transmembrane currents
    and calculating extracellular potential ECP
    """

    def __init__(self, cell: StylizedCell, electrode_positions: Optional[Union[List[int], np.ndarray]],
                 move_cell: Optional[Union[Tuple[List[float], Tuple[float, float, float]]]] = None,
                 scale: float = 1.0, min_distance: Optional[float] = None) -> None:
        """
        cell: cell object
        electrode_positions: n-by-3 array of electrodes coordinates
        move_cell, scale, min_distance: see method 'calc_transfer_resistance'
        """
        self.tr = None
        self.cell = cell
        self.elec_coords = np.array(electrode_positions)
        if self.elec_coords.ndim != 2 or self.elec_coords.shape[1] != 3:
            raise ValueError("electrode_positions must be an n-by-3 2-D array")
        self.nelec = self.elec_coords.shape[0]
        self.im_rec = self.__record_im()
        # self.cell.calc_seg_coords()
        self.calc_transfer_resistance(move_cell, scale, min_distance)

    # PRIVATE METHODS
    def __record_im(self) -> Recorder:
        """Enable extracellular mechanism in Neuron and record transmembrane currents"""
        h.cvode.use_fast_imem(1)
        for sec in self.cell.all:
            sec.insert('extracellular')  # insert extracellular

        for inj in self.cell.injection:
            if inj.rec_vec is None:
                inj.setup_recorder()
        return Recorder(self.cell.segments, 'i_membrane_')

    # PUBLIC METHODS
    def calc_transfer_resistance(self, move_cell: Optional[Union[Tuple[List[float], Tuple[float, float, float]]]] = None,
                                 scale: float = 1.0, min_distance: Optional[float] = None) -> None:
        """
        Precompute mapping from segment to electrode locations
        move_cell: tuple of (translate,rotate), rotate the cell followed by translating it
        scale: scaling factor of ECP magnitude
        min_distance: minimum distance allowed between segment and electrode, if specified
        """
        sigma = 0.3  # mS/mm
        seg_coords = self.cell.seg_coords
        if move_cell is None:
            dl = seg_coords['dl']
            pc = seg_coords['pc']
        else:
            dl = move_position([0., 0., 0.], move_cell[1], seg_coords['dl'])
            pc = move_position(move_cell[0], move_cell[1], seg_coords['pc'])
        if min_distance is None:
            r = seg_coords['r']
        else:
            r = np.fmax(seg_coords['r'], min_distance)
        tr = np.empty((self.nelec, self.cell._nseg))
        for j in range(self.nelec):  # calculate mapping for each site on the electrode
            rel_pc = self.elec_coords[j, :] - pc  # distance between electrode and segment centers
            # compute dot product row-wise, the resulting array has as many rows as original
            r2 = np.einsum('ij,ij->i', rel_pc, rel_pc)
            rlldl = np.einsum('ij,ij->i', rel_pc, dl)
            dlmag = np.linalg.norm(dl, axis=1)  # length of each segment
            rll = abs(rlldl / dlmag)  # component of r parallel to the segment axis it must be always positive
            r_t2 = r2 - rll ** 2  # square of perpendicular component
            up = rll + dlmag / 2
            low = rll - dlmag / 2
            np.fmax(r_t2, r ** 2, out=r_t2, where=low - r < 0)
            num = up + np.sqrt(up ** 2 + r_t2)
            den = low + np.sqrt(low ** 2 + r_t2)
            tr[j, :] = np.log(num / den) / dlmag  # units of (um) use with im_ (total seg current)
        self.tr = scale / (4 * np.pi * sigma) * tr

    def calc_ecp(self) -> np.ndarray:
        """Calculate ECP after simulation. Unit: mV."""
        im = self.im_rec.as_numpy()
        for inj in self.cell.injection:
            im[inj.get_segment_id(), :] -= inj.rec_vec.as_numpy()
        ecp = np.dot(self.tr, im)  # im unit nA, ecp unit mV
        return ecp


def move_position(translate: List[float], rotate: Tuple[float, float, float],
                  old_position: Optional[Union[List[float], np.ndarray]] = None, move_frame: bool = False) -> np.ndarray:
    """
    Rotate and translate an object with old_position and calculate its new coordinates. Rotate(alpha,h,phi): first
    rotate alpha about y-axis (spin), then rotate arccos(h) about x-axis (elevation), then rotate phi about y axis (
    azimuth). Finally translate the object by translate(x,y,z). If move_frame is True, use the object as reference
    frame and move the old reference frame, calculate new coordinates of the old_position.
    """
    translate = np.asarray(translate)
    if old_position is None:
        old_position = [0., 0., 0.]
    old_position = np.asarray(old_position)
    rot = Rotation.from_euler('yxy', [rotate[0], np.arccos(rotate[1]), rotate[2]])
    if move_frame:
        new_position = rot.inv().apply(old_position - translate)
    else:
        new_position = rot.apply(old_position) + translate
    return new_position
