import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

import h5py

from stylized_module.base import inference as infer
import config.paths as paths

if __name__ == "__main__":
    inf = infer.Inferencer()
    # f = h5py.File(paths.SIMULATIONS, 'w')
    # dset = f.create_dataset("input", (inf.simR.sim.n,1+len(inf.simR.sim.geometry)+6))
    # dset[:,:] = inf.simR.sim.input_array
    # f.close()
    # theta, x = inf.simR.simulate_in_sbi(inf, inf.prior)
    # posterior = inf.run_inferencer(theta, x, inf.prior)
    # theta, x = sc.simulate_in_sbi(inf, posterior)
    # posterior = inf.run_inferencer(theta, x, posterior)
    # samples, log_prob = inf.build_log_prob(posterior)
    # predicted_params = inf.predict_params(samples, log_prob)