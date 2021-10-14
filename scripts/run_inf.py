import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

import numpy as np
import torch
import pickle

import stylized_module.base.inference as infer
import config.params as params
import config.paths as paths


if __name__ == "__main__":
    with open("posterior_mdn2.pkl", "rb") as handle:
        proposal = pickle.load(handle)
    theta = torch.tensor(np.load("theta_mdn3.npy"))
    x = torch.tensor(np.load("x_mdn3.npy"))
    print(theta.shape, x.shape)
    inf = infer.Inferencer(proposal)
#    inf = infer.Inferencer(params.IM_PRIOR_DISTRIBUTION)
    posterior = inf.run_inferencer(theta, x, inf.prior)

    with open("posterior_mdn3.pkl", "wb") as handle:
        pickle.dump(posterior, handle)
