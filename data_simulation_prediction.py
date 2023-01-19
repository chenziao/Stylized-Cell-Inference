from neuron import h
import pandas as pd
import numpy as np
import json
import os, sys
import time
import datetime
from tqdm import tqdm
import argparse

from cell_inference.config import paths, params
from cell_inference.cells.simulation import SIMULATION_CLASS
from cell_inference.cells.stylizedcell import CellTypes
from cell_inference.utils.random_parameter_generator import generate_predicted_parameters_from_config
from cell_inference.utils.feature_extractors.SummaryStats2D import process_lfp
from cell_inference.utils.data_manager import NpzFilesCollector

TRIAL_NAME = 'Reduced_Order_stochastic_spkwid_trunkLR4_LactvCa_Loc5_restrict_h' # select trial
MODEL_NAME = 'CNN_batch256_dv' # select model
INVIVO_NAME = 'all_cell_LFP2D_Analysis_SensorimotorSpikeWaveforms_NP_SUTempFilter_NPExample' # select data

def set_path(trial=None, model=None, invivo=None):
    global TRIAL_NAME, TRIAL_PATH, MODEL_NAME, MODEL_PATH, INVIVO_NAME, INVIVO_PRED_PATH
    global PRED_PATH, PRED_LFP_PATH, PRED_STATS_PATH
    if trial is not None:
        TRIAL_NAME = trial
        TRIAL_PATH = os.path.join(paths.SIMULATED_DATA_PATH, TRIAL_NAME)
    if model is not None:
        MODEL_NAME = model
        MODEL_PATH = os.path.join(TRIAL_PATH, MODEL_NAME)
    if invivo is not None:
        INVIVO_NAME = invivo
        INVIVO_PRED_PATH = os.path.join(MODEL_PATH, INVIVO_NAME)
    PRED_PATH = os.path.join(INVIVO_PRED_PATH, MODEL_NAME + '_prediction.csv')
    PRED_LFP_PATH = os.path.join(INVIVO_PRED_PATH, 'lfp_' + MODEL_NAME + '_pred.npz')  # LFP and labels
    PRED_STATS_PATH = os.path.join(INVIVO_PRED_PATH, 'summ_stats_' + MODEL_NAME + '_pred.npz')  # summary statistics

set_path(TRIAL_NAME, MODEL_NAME, INVIVO_NAME)

def load_pred_data(config_dict=None):
    if config_dict is None:
        CONFIG_PATH = os.path.join(TRIAL_PATH, 'config.json')  # trial configuration
        with open(CONFIG_PATH, 'r') as f:
            config_dict = json.load(f)
    pred_dict = pd.read_csv(PRED_PATH, index_col=0)
    pred_dict = pred_dict.to_dict(orient='list')
    pred_dict = {key: np.array(value) for key, value in pred_dict.items()}
    return config_dict, pred_dict

def run_pred_simulation(config_dict, pred_dict, number_locs = 3,
                        batch_id=None, number_cells_per_batch=None,
                        save_stats=False, rand_seed=None):
    ## Initialize Neuron
    h.load_file('stdrun.hoc')
    try:
        h.nrn_load_dll(paths.COMPILED_LIBRARY_REDUCED_ORDER)
    except:
        pass
    h.dt = params.DT
    h.steps_per_ms = 1/h.dt
    geo_standard = pd.read_csv(paths.GEO_REDUCED_ORDER, index_col='id')

    ## Select cells for batch
    number_cells = list(pred_dict.values())[0].size
    if batch_id is None or number_cells_per_batch is None:
        batch_id = 0
        batch_suf = ''
        number_cells_per_batch = number_cells
    else:
        batch_suf = '_%d' % batch_id
    if batch_id * number_cells_per_batch >= number_cells:
        exit_msg = 'Batch ID outside the range of number of cells.'
        if __name__ == "__main__":
            sys.exit(exit_msg)
        else:
            raise ValueError(exit_msg)
    cell_index = slice(batch_id * number_cells_per_batch, (batch_id + 1) * number_cells_per_batch)

    ## Set up simulation parameters
    inference_list = config_dict['Trial_Parameters']['inference_list']
    rand_seed = config_dict['Trial_Parameters']['rand_seed'] if rand_seed is None else int(rand_seed)
    sim_param = config_dict['Simulation_Parameters']
    syn_params = {}

    # Simulation parameters
    simulation_class = sim_param.get('simulation_class', 'Simulation')
    h.tstop = sim_param.get('tstop', params.DT)

    # Synapse parameters
    if sim_param.get('gmax_mapping') is None:
        syn_params['gmax'] = sim_param.get('gmax')
    else:
        pass # TODO
    syn_params['syn_sec'] = sim_param.get('syn_sec', 0)
    syn_params['syn_loc'] = sim_param.get('syn_loc', .5)
    syn_params['stim_param'] = sim_param.get('stim_param', {})

    syn_params['tstart'] = sim_param.get('tstart')
    syn_params['point_conductance_division'] = sim_param.get('point_conductance_division')
    syn_params['dens_params'] = sim_param.get('dens_params')
    syn_params['cnst_params'] = sim_param.get('cnst_params')
    syn_params['has_nmda'] = sim_param.get('has_nmda', True)

    # Biophysical parameters
    filepath = sim_param.get('full_biophys')
    with open(filepath) as f:
        full_biophys = json.load(f)

    # Common parameters
    biophys_param = sim_param.get('biophys_param', [])
    biophys_comm = sim_param.get('biophys_comm', {})

    # Whether use parameter interpreter
    interpret_params = sim_param.get('interpret_params', False)
    interpret_type = sim_param.get('interpret_type', 0)

    gen_params = generate_predicted_parameters_from_config(config_dict, pred_dict, number_locs=number_locs)
    labels, rand_param, loc_param, geo_param = [x[cell_index] for x in gen_params]
    number_cells = labels.shape[0]
    print(loc_param.shape)
    print(geo_param.shape)
    print(rand_param.shape)
    print(labels.shape)

    ## Run simulation
    timer_start = time.time()
    sim = SIMULATION_CLASS[simulation_class](
        cell_type = CellTypes.REDUCED_ORDER,
        ncell = number_cells,
        geometry = geo_standard,
        electrodes = params.ELECTRODE_POSITION,
        loc_param = loc_param,
        geo_param = geo_param,
        biophys = biophys_param,
        full_biophys = full_biophys,
        biophys_comm = biophys_comm,
        interpret_params = interpret_params,
        interpret_type = interpret_type,
        min_distance = params.MIN_DISTANCE,
        record_soma_v = False,
        spike_threshold = params.SPIKE_THRESHOLD,
        randseed = rand_seed,
        **syn_params
    )
    sim.run_neuron_sim()
    print('Simulation run time: ' + str(datetime.timedelta(seconds=time.time() - timer_start)))

    ## Process simulation results
    # Remove cells with invalid firing pattern
    if simulation_class == 'Simulation_stochastic':
        spk_windows, nspk = sim.get_spk_windows('all')
        valid = nspk > 1
        firing_rate = 1000. * nspk / (h.tstop - sim.tstart * h.dt)
    else:
        nspk, _ = sim.get_spike_number('all')
        valid = nspk == 1

    invalid = np.nonzero(~valid)[0]
    valid = np.nonzero(valid)[0]
    invalid_nspk = nspk[invalid]
    for n in np.unique(invalid_nspk):
        print('%d cells fire %d times.' % (np.count_nonzero(invalid_nspk==n), n))

    invalid_params = {}
    invalid_params['geo_param'] = geo_param[invalid, :]
    invalid_params['gmax'] = None if sim.gmax is None else sim.gmax[invalid]

    number_samples = valid.size
    labels = np.delete(labels, invalid, axis=0)
    rand_param = np.delete(rand_param, invalid, axis=0)
    gmax = None if sim.gmax is None else sim.gmax[valid] 

    additional_save = {}
    if simulation_class == 'Simulation_stochastic':
        additional_save['firing_rate'] = firing_rate[valid]

    # Get LFP for valid cells
    if simulation_class == 'Simulation_stochastic':
        lfp_cell = lambda i: np.mean(sim.get_eaps_by_windows(index=valid[i],
            spk_windows=spk_windows[valid[i]], multiple_position=True), axis=0)
    else:
        timer_start = time.time()
        start_idx = int(max(np.ceil(sim.stim.start / h.dt) - params.PK_TR_IDX_IN_WINDOW, 0)) # ignore signal before
        lfp = sim.get_lfp(index=valid, t_index=slice(start_idx, None), multiple_position=True) # (cells x locs x channels x time)
        lfp = np.moveaxis(np.mean(lfp, axis=1), -2, -1) # -> (cells x channels x time) -> (samples x time x channels)
        print('LFP run time: ' + str(datetime.timedelta(seconds=time.time() - timer_start)))
        lfp_cell = lambda i: lfp[i]

    # Process LFP
    pad_spike_window = True
    bad_cases = tuple(range(-1,3)) if pad_spike_window else tuple(range(3))
    if 'y' in inference_list:
        y_idx = inference_list.index('y')
        ycoord = lambda i: labels[i, y_idx]
    else:
        ycoord = lambda i: None

    bad_indices = {bad: [] for bad in bad_cases}
    lfp_list = []
    yshift = []
    summ_stats = []
    good_count = 0

    for i in tqdm(range(number_samples)):
        bad, g_lfp, _, _, _, ys, ss = process_lfp(
            lfp_cell(i), dt=None, pad_spike_window=pad_spike_window, ycoord=ycoord(i),
            gauss_filt=True, calc_summ_stats=save_stats, additional_stats=1, err_msg=True
        )
        bad_indices[bad].append(i)
        if bad<=0:
            good_count += 1
            lfp_list.append(g_lfp)
            yshift.append(ys)
            if save_stats:
                summ_stats.append(ss)

    t = sim.t()[:params.WINDOW_SIZE]
    windowed_lfp = np.stack(lfp_list, axis=0)  # (samples x time window x channels)
    yshift = np.array(yshift)
    summ_stats = np.array(summ_stats)

    good_indices = np.sort([i for bad, indices in bad_indices.items() if bad<=0 for i in indices])
    print('%d good samples out of %d samples.' % (good_count, number_samples))
    for bad, indices in bad_indices.items():
        print('Bad case %d bad: %d samples.' % (bad, len(indices)))

    # Save result data
    np.savez(add_batch_id(PRED_LFP_PATH, batch_suf), t=t, x=windowed_lfp, y=labels, ys=yshift,
             rand_param=rand_param, gmax=gmax, bad_indices=bad_indices, good_indices=good_indices,
             invalid_params=invalid_params, valid=valid, invalid=invalid, **additional_save)
    if save_stats:
        np.savez(add_batch_id(PRED_STATS_PATH, batch_suf), x=summ_stats, y=labels[good_indices], ys=yshift,
                 rand_param=rand_param[good_indices], gmax=None if gmax is None else gmax[good_indices])

def add_batch_id(filename, batch_suf):
    fname, fext = os.path.splitext(filename)
    return ''.join((fname, batch_suf, fext))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-trial', type=str, nargs='?', default=TRIAL_NAME, help="Trial name", metavar='Trial')
    parser.add_argument('-model', type=str, nargs='?', default=MODEL_NAME, help="Model name", metavar='Model')
    parser.add_argument('-invivo', type=str, nargs='?', default=INVIVO_NAME, help="In vivo data name", metavar='Invivo')
    parser.add_argument('batch_id', type=int, nargs='?', default=None, help="Batch ID", metavar='Batch ID')
    parser.add_argument('-c', type=int, nargs='?', default=None, help="Number of cells per batch", metavar='# Cells')
    parser.add_argument('-l', type=int, nargs='?', default=3, help="Number of locations per cell", metavar='# Locations')
    parser.add_argument('-seed', type=int, nargs='?', default=None, help="Random seed", metavar='Seed')
    parser.add_argument('--stats', action='store_true', help="Save summary statistics")
    parser.add_argument('--no-stats', action='store_false', dest='stats', help="Do not save summary statistics")
    parser.set_defaults(stats=False)
    
    args = parser.parse_args()

    set_path(trial=args.trial, model=args.model, invivo=args.invivo)
    config_dict, pred_dict = load_pred_data()
    run_pred_simulation(config_dict, pred_dict, number_locs = args.l,
                        batch_id=args.batch_id, number_cells_per_batch=args.c,
                        save_stats=args.stats, rand_seed=args.seed)
