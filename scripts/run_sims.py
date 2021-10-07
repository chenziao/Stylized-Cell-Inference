import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

import h5py
import pandas as pd

from neuron import h

import stylized_module.base.inference as infer
import config.paths as paths
from utils.combine_csv import build_csv

pc = h.ParallelContext()
MPI_size = int(pc.nhost())
MPI_rank = int(pc.id())

if __name__ == "__main__":
    inf = infer.Inferencer()
    # f = h5py.File(paths.SIMULATIONS, 'w')
    # if MPI_rank == 0:
    #     build_csv()
    theta, x = inf.simR.simulate_runs(inf.prior)
    print(theta.shape, x.shape)
    posterior = inf.run_inferencer(theta, x, inf.prior)
    pc.done()
    # posterior = inf.run_inferencer(theta, x, posterior)
    # samples, log_prob = inf.build_log_prob(posterior)
    # predicted_params = inf.predict_params(samples, log_prob)