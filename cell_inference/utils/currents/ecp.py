from neuron import h
import numpy as np
from scipy.spatial.transform import Rotation
import h5py
from typing import Optional, List, Tuple, Union

from cell_inference.utils.currents.recorder import Recorder
from cell_inference.cells.stylizedcell import StylizedCell

class EcpCell(object):
    def __init__(self, file: Optional[str] = None,
                 im: Optional[np.ndarray] = None,
                 seg_coords: Optional[dict] = None) -> None:
        if file is None:
            self.im_rec = self.Im_rec(im)
            self.seg_coords = seg_coords
        else:
            self.load_from_file(file)
        self.im = self.im_rec.as_numpy()
        self._nseg = self.im.shape[0]
        self.injection = []

    def load_from_file(self, file):
        with h5py.File(file,'r') as hf:
            seg_coords = {}
            for key in hf['seg_coords']:
                seg_coords[key] = hf['seg_coords'][key][()]
            self.im_rec = self.Im_rec(hf['data'])
            self.seg_coords = seg_coords

    class Im_rec(object):
        def __init__(self, im):
            self.im = np.array(im)

        def as_numpy(self):
            return self.im

 
class EcpMod(object):
    """
    A module for recording single cell transmembrane currents
    and calculating extracellular potential ECP
    """

    def __init__(self, cell: Union[StylizedCell, EcpCell],
                 electrode_positions: Union[List[List], np.ndarray] = [[0., 0., 0.]],
                 move_cell: Optional[Union[List,Tuple]] = None,
                 scale: float = 1.0, min_distance: Optional[float] = None) -> None:
        """
        cell: cell object
        electrode_positions: n-by-3 array of electrodes coordinates
        move_cell, scale, min_distance: see method 'calc_transfer_resistance'
        """
        self.cell = cell
        self.set_electrode_positions(electrode_positions)
        self.move_cell = move_cell
        self.scale = scale
        self.min_distance = min_distance
        self.ecp_cell = isinstance(cell, EcpCell)
        self.__record_im()

    # PRIVATE METHODS
    def __record_im(self) -> Recorder:
        """Enable extracellular mechanism in Neuron and record transmembrane currents"""
        if self.ecp_cell:
            self.im_rec = self.cell.im_rec
        else:
            h.cvode.use_fast_imem(1)
            for sec in self.cell.all:
                sec.insert('extracellular')  # insert extracellular
            for inj in self.cell.injection:
                if inj.rec_vec is None:
                    inj.setup_recorder()
            self.im_rec = Recorder(self.cell.segments, 'i_membrane_')

    # PUBLIC METHODS
    def set_electrode_positions(self, electrode_positions: Union[List[List], np.ndarray]) -> None:
        self.elec_coords = np.asarray(electrode_positions)
        if self.elec_coords.ndim != 2 or self.elec_coords.shape[1] != 3:
            raise ValueError("'electrode_positions' must be an n-by-3 2-D array")
        self.nelec = self.elec_coords.shape[0]
    
    def calc_transfer_resistance(self, move_cell: Optional[Union[List,Tuple]] = None,
                                 scale: float = 1.0, min_distance: Optional[float] = None,
                                 move_elec: Optional[bool] = False, sigma: float = 0.3,) -> None:
        """
        Precompute mapping from segment to electrode locations
        move_cell: list/tuple/2-by-3 array of (translate,rotate), rotate the cell followed by translating it
        scale: scaling factor of ECP magnitude
        min_distance: minimum distance allowed between segment and electrode, if specified
        sigma: resistivity of medium (mS/mm)
        move_elec: whether or not to relatively move electrodes for calculation
        """
        seg_coords = self.cell.seg_coords
        if move_cell is not None:
            move_cell = np.asarray(move_cell).reshape((2, 3))
        if move_elec and move_cell is not None:
            elec_coords = move_position(move_cell[0], move_cell[1], self.elec_coords, True)
        else:
            elec_coords = self.elec_coords
        if not move_elec and move_cell is not None:
            dl = move_position([0., 0., 0.], move_cell[1], seg_coords['dl'])
            pc = move_position(move_cell[0], move_cell[1], seg_coords['pc'])
        else:
            dl = seg_coords['dl']
            pc = seg_coords['pc']
        if min_distance is None:
            r = seg_coords['r']
        else:
            r = np.fmax(seg_coords['r'], min_distance)
        rr = r ** 2
        
        tr = np.empty((self.nelec, self.cell._nseg))
        for j in range(self.nelec):  # calculate mapping for each site on the electrode
            rel_pc = elec_coords[j, :] - pc  # distance between electrode and segment centers
            # compute dot product row-wise, the resulting array has as many rows as original
            r2 = np.einsum('ij,ij->i', rel_pc, rel_pc)
            rlldl = np.einsum('ij,ij->i', rel_pc, dl)
            dlmag = np.linalg.norm(dl, axis=1)  # length of each segment
            rll = abs(rlldl / dlmag)  # component of r parallel to the segment axis it must be always positive
            r_t2 = r2 - rll ** 2  # square of perpendicular component
            up = rll + dlmag / 2
            low = rll - dlmag / 2
            np.fmax(r_t2, rr, out=r_t2, where=low - r < 0)
            num = up + np.sqrt(up ** 2 + r_t2)
            den = low + np.sqrt(low ** 2 + r_t2)
            tr[j, :] = np.log(num / den) / dlmag  # units of (um) use with im_ (total seg current)
        tr *= scale / (4 * np.pi * sigma)
        return tr

    def calc_im(self) -> np.ndarray:
        """Calculate transmembrane current after simulation. Unit: nA."""
        im = self.im_rec.as_numpy()
        for inj in self.cell.injection:
            im[inj.get_segment_id(), :] -= inj.rec_vec.as_numpy()
        return im

    def calc_ecp(self, **kwargs) -> np.ndarray:
        """Calculate ECP after simulation. Unit: mV."""
        kwargs0 = {
                    'move_cell': self.move_cell,
                    'scale': self.scale,
                    'min_distance': self.min_distance,
                   }
        for key, value in kwargs.items():
            kwargs0[key] = value
        tr = self.calc_transfer_resistance(**kwargs0)
        im = self.calc_im()
        return tr @ im

    def calc_ecps(self, move_cell: Optional[List] = None, **kwargs) -> np.ndarray:
        """Calculate ECP with multiple positions after simulation. Unit: mV."""
        kwargs0 = {
                    'scale': self.scale,
                    'min_distance': self.min_distance,
                   }
        for key, value in kwargs.items():
            kwargs0[key] = value
        if move_cell is None:
            move_cell = [self.move_cell]
        im = self.calc_im()
        ecp = []
        for mc in move_cell:
            ecp.append(self.calc_transfer_resistance(move_cell=mc, **kwargs0) @ im)
        return np.stack(ecp, axis=0)


def move_position(translate: Union[List[float],Tuple[float],np.ndarray],
                  rotate: Union[List[float],Tuple[float],np.ndarray],
                  old_position: Optional[Union[List[float], np.ndarray]] = None,
                  move_frame: bool = False) -> np.ndarray:
    """
    Rotate and translate an object with old_position and calculate its new coordinates.
    Rotate(alpha, h, phi): first rotate alpha about the y-axis (spin),
    then rotate arccos(h) about the x-axis (elevation),
    then rotate phi about the y-axis (azimuth).
    Finally translate the object by translate(x, y, z).
    If move_frame is True, use the object as reference frame and move the
    old reference frame, calculate new coordinates of the old_position.
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
