import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

import pickle
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn  # For SNLE: likelihood_nn(). For SNRE: classifier_nn()

import config.params as params
import config.paths as paths
from stylized_module.base.simulate_cells import run_pm_simulation, run_am_simulation
from stylized_module.models.SummaryStats2D import cat_output


class Inferencer(object):

    def __init__(self):
        self.sim, self.window_size, self.x0_trace, self.t0 = run_pm_simulation() if params.ACTIVE_CELL is False else run_am_simulation()
        self.fst_idx = first_pk_tr(x0_trace)
        self.simulator, self.prior = prepare_for_sbi(simulate, params.IM_PRIOR_DISTRIBUTION)
        self.x_o = cat_output(x0_trace)

        density_estimator_build_fun = posterior_nn(model=params.IM_POSTERIOR_MODEL_ESTIMATOR,
                                           embedding_net=params.IM_EMBEDDED_NETWORK,
                                           hidden_features=params.IM_POSTERIOR_MODEL_HIDDEN_LAYERS)

        self.inference = SNPE(prior=prior,density_estimator=density_estimator_build_fun,show_progress_bars=True)


    def run_inferencer(self):
        posteriors = []
        proposal = self.prior

        for i in range(params.IM_NUMBER_OF_ROUNDS):
            theta, x = simulate_for_sbi(self.simulator,proposal,num_simulations=params.IM_NUMBER_OF_SIMULATIONS)
            # In `SNLE` and `SNRE`, you should not pass the `proposal` to `.append_simulations()`
        #     density_estimator = inference.append_simulations(np.squeeze(theta), np.squeeze(x), proposal=proposal).train()
            density_estimator = self.inference.append_simulations(np.squeeze(theta), x, proposal=proposal).train()
            posterior = self.inference.build_posterior(density_estimator, sample_with="mcmc")
            
            with open(paths.POSTERIOR_SAVE + str(i) + "_post.pkl", "wb") as handle:
                pickle.dump(posterior, handle)
                
            with open(paths.POSTERIOR_SAVE + str(i) + "_de.pkl", "wb") as handle:
                pickle.dump(density_estimator, handle)
                
            posteriors.append(posterior)
            proposal = posterior.set_default_x(x_o)
            
        inference._summary_writer = None
        inference._build_neural_net = None
        with open(paths.INFERENCER_SAVE + str(i) + ".pkl", "wb") as handle:
            pickle.dump(inference, handle)

        # with open(paths.POSTERIOR_SAVE + "1_post.pkl", "rb") as handle:
        #     posterior = pickle.load(handle)