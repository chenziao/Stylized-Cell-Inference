import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

import h5py
import pandas as pd

import stylized_module.base.inference as infer
import config.paths as paths
from utils.combine_csv import build_csv

if __name__ == "__main__":
    inf = infer.Inferencer()
    # f = h5py.File(paths.SIMULATIONS, 'w')
    build_csv()
    theta, x = inf.simR.simulate_in_sbi(inf.prior)
    print(theta.shape, x.shape)
    posterior = inf.run_inferencer(theta, x, inf.prior)
    # theta, x = sc.simulate_in_sbi(inf, posterior)
    # posterior = inf.run_inferencer(theta, x, posterior)
    # samples, log_prob = inf.build_log_prob(posterior)
    # predicted_params = inf.predict_params(samples, log_prob)