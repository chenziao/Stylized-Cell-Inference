import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

import pickle
from sbi.inference import SNPE, prepare_for_sbi
from sbi.utils.get_nn_models import posterior_nn  # For SNLE: likelihood_nn(). For SNRE: classifier_nn()
from utils.spike_window import first_pk_tr
import torch
import numpy as np

import config.params as params
import config.paths as paths
from stylized_module.base.simulate_cells import SimulationRunner #run_pm_simulation, run_am_simulation, simulate
from stylized_module.models.SummaryStats2D import cat_output
from utils.transform.distribution_transformation import norm2unif, range2logn, norm2logn, logds_norm2unif, logds_norm2logn


class Inferencer(object):

    def __init__(self):
        # self.sim, self.window_size, self.x0_trace, self.t0 = run_pm_simulation() if params.ACTIVE_CELL is False else run_am_simulation()
        self.simR = SimulationRunner()
        self.fst_idx = first_pk_tr(self.simR.x0_trace)
        self.simulator, self.prior = prepare_for_sbi(self.simR.simulate, params.IM_PRIOR_DISTRIBUTION)
        self.x_o = cat_output(self.simR.x0_trace)

        density_estimator_build_fun = posterior_nn(model=params.IM_POSTERIOR_MODEL_ESTIMATOR,
                                           embedding_net=params.IM_EMBEDDED_NETWORK,
                                           hidden_features=params.IM_POSTERIOR_MODEL_HIDDEN_LAYERS)

        self.inference = SNPE(prior=self.prior,density_estimator=density_estimator_build_fun,show_progress_bars=True)


    def run_inferencer(self, theta, x, proposal):
        # posteriors = []
        # proposal = self.prior

        # for i in range(params.IM_NUMBER_OF_ROUNDS):
        # theta, x = simulate_for_sbi(self.simulator,proposal,num_simulations=params.IM_NUMBER_OF_SIMULATIONS)
        # In `SNLE` and `SNRE`, you should not pass the `proposal` to `.append_simulations()`
    #     density_estimator = inference.append_simulations(np.squeeze(theta), np.squeeze(x), proposal=proposal).train()
        density_estimator = self.inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = self.inference.build_posterior(density_estimator, sample_with="mcmc")
        
        with open(paths.POSTERIOR_SAVE + "_post.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
            
        with open(paths.POSTERIOR_SAVE + "_de.pkl", "wb") as handle:
            pickle.dump(density_estimator, handle)
            
        # posteriors.append(posterior)
        proposal = posterior.set_default_x(self.x_o)
            
        self.inference._summary_writer = None
        self.inference._build_neural_net = None
        with open(paths.INFERENCER_SAVE + ".pkl", "wb") as handle:
            pickle.dump(self.inference, handle)

        # with open(paths.POSTERIOR_SAVE + "1_post.pkl", "rb") as handle:
        #     posterior = pickle.load(handle)
        return posterior

    def build_log_prob(self, posterior):
        samples = posterior.sample((1000,), x=self.x_o, sample_with='mcmc') #, sample_with_mcmc=True

        #posterior.leakage_correction(x_o, num_rejection_samples=1000)
        log_probability = posterior.log_prob(samples,x=self.x_o, norm_posterior=False) #, norm_posterior=False
        log_prob_t = log_probability
        for i in range(6):
            log_prob_t += logds_norm2unif(samples[:,i], params.IM_PARAMETER_BOUNDS[i][0], params.IM_PARAMETER_BOUNDS[i][1])
        for i in range(6,11):
            if i == 6:
                m,s=range2logn(params.IM_PARAMETER_BOUNDS[i][0], params.IM_PARAMETER_BOUNDS[i][1], n_sigma=3)
            else:
                m,s=range2logn(params.IM_PARAMETER_BOUNDS[i][0], params.IM_PARAMETER_BOUNDS[i][1])
            log_prob_t += logds_norm2logn(samples[:,i], m, s)
        return samples, log_prob_t

    def predict_params(self, samples, log_probability):
        sample_idx = np.argmax(log_probability)
        samples_t = torch.clone(samples)
        for i in range(6):
            samples_t[:,i] = torch.from_numpy(norm2unif(samples[:,i], params.IM_PARAMETER_BOUNDS[i][0], params.IM_PARAMETER_BOUNDS[i][1]))
        for i in range(6,11):
            if i == 6:
                m,s=range2logn(params.IM_PARAMETER_BOUNDS[i][0], params.IM_PARAMETER_BOUNDS[i][1], n_sigma=3)
            else:
                m,s=range2logn(params.IM_PARAMETER_BOUNDS[i][0], params.IM_PARAMETER_BOUNDS[i][1])
            samples_t[:,i] = norm2logn(samples[:,i], m, s)
        predicted_post = samples[sample_idx]
        return predicted_post