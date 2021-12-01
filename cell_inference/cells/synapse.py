# Dependencies
from neuron import h

# Project Imports
from cell_inference.cells.stylizedcell import StylizedCell
from cell_inference.utils.currents.pointcurrent import PointCurrent


class Synapse(PointCurrent):
    def __init__(self, cell: StylizedCell, stim: h.NetStim, sec_index: int,
                 gmax: float = 0.01, loc: float = 0.5, record: bool = True) -> None:
        super().__init__(cell, sec_index, loc)
        self.stim = stim
        self.gmax = gmax
        self.pp_obj = h.AlphaSynapse1(self.get_section()(loc))
        self.setup(record)
        self.nc = self.__setup_synapse()

    # PRIVATE METHODS
    def __setup_synapse(self) -> h.NetCon:
        syn = self.pp_obj
        syn.e = 0.  # mV. Reversal potential
        syn.tau = 2.0  # ms. Synapse time constant
        syn.gmax = self.gmax  # uS. maximum conductance
        return h.NetCon(self.stim, syn, 1, 0, 1)

    # PUBLIC METHODS
    def setup(self, record: bool = True) -> None:
        self.__setup_synapse()
        if record:
            self.setup_recorder()
