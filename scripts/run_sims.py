import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

import h5py
import pandas as pd
import numpy as np

import _pickle as cPickle
from neuron import h

import stylized_module.base.inference as infer
import config.paths as paths
from utils.combine_csv import build_lfp_csv

pc = h.ParallelContext()
MPI_size = int(pc.nhost())
MPI_rank = int(pc.id())

if __name__ == "__main__":
    #inf = infer.Inferencer()
    # f = h5py.File(paths.SIMULATIONS, 'w')
    # if MPI_rank == 0:
    #     build_csv()
    with open(r"posterior.pkl", "rb") as input_file:
        posterior = cPickle.load(input_file)
    inf = infer.Inferencer()
    theta, x = inf.simR.simulate_runs(posterior)
    pc.done()
    print(theta.shape)
    print(x.shape)
    np.save("theta2.npy", theta)
    np.save("x2.npy", x)
    # pc.barrier()
    # pc.done()
    # if MPI_rank == 0:
    #     build_lfp_csv()
    # # print(theta.shape, x.shape)
    # posterior = inf.run_inferencer(theta, x, inf.prior)
    # theta, x = inf.simR.simulate_runs(posterior)
    # # pc.done()
    # posterior = inf.run_inferencer(theta, x, posterior)
    # samples, log_prob = inf.build_log_prob(posterior)
    # predicted_params = inf.predict_params(samples, log_prob)
