from neuron import h

from cell_inference.config import paths, params
from data_simulation import DataSimulator
from train_model import Trainer
from tqdm import tqdm

if __name__ == '__main__':
    print("First Execution")
    h.load_file('stdrun.hoc')
    h.nrn_load_dll(paths.COMPILED_LIBRARY)
    h.tstop = params.TSTOP
    h.dt = params.DT
    inference_list = ['y', 'd', 'theta', 'h', 'phi', 'r_s', 'l_t', 'r_t']
    number_samples = 1000
    data_path = '1000s_y1Loc2Alt_Ori2_Geo3_params'
    iteration = 20
    # for i in tqdm(range(iteration)):
    #     ds = DataSimulator(inference_list=inference_list, number_samples=number_samples, random_seed=i)
    #     ds.simulate_params(data_path=data_path, iteration=iteration)
    print("Starting Training")
    tr = Trainer(trial_path=data_path)
    tr.normalize_labels()
    model = tr.build_data_and_fit()
    print("Finished")



