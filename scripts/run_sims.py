import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

from stylized_module.base import simulate_cells as sc
from stylized_module.base import inference as infer


if __name__ == "__main__":
    inf = infer.Inferencer()
    theta, x = sc.simulate_in_sbi(inf, inf.prior)
    # posterior = inf.run_inferencer(theta, x, inf.prior)
    # theta, x = sc.simulate_in_sbi(inf, posterior)
    # posterior = inf.run_inferencer(theta, x, posterior)
    # samples, log_prob = inf.build_log_prob(posterior)
    # predicted_params = inf.predict_params(samples, log_prob)