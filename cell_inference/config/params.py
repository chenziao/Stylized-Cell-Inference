import numpy as np
import h5py
import cell_inference.config.paths as paths

ACTIVE_CELL = False

# GENERAL PARAMETERS USED ACROSS RUNS
TSTOP = 20.  # ms
DT = 0.025  # ms. Change with h.steps_per_ms = 1/h.dt
STIM_PARAM = {'start': 2.}
SPIKE_THRESHOLD = -30.

# ELECTRODE_POSITION = np.column_stack((np.zeros(96),np.linspace(-1900,1900,96),np.zeros(96)))
hf = h5py.File(paths.ELECTRODES, 'r')
elec_pos = np.array(hf.get('coord'))
ELECTRODE_POSITION = np.column_stack((elec_pos, np.zeros(elec_pos.shape[0])))
ELECTRODE_GRID = (np.array(hf.get('grid/x')), np.array(hf.get('grid/y')), np.zeros(1))

# GET GROUND TRUTH FROM ACTIVE MODEL PARAMS ARE DENOTED WITH THE PREFIX GT
# x, y, z, alpha, h, phi
LOCATION_PARAMETERS = [0., 0., 50., np.pi / 4, 1., 0.]
GMAX = 0.005

# LFP PROCESSING
MIN_DISTANCE = 25.0

BUTTERWORTH_ORDER = 2  # 2nd order
FILTER_CRITICAL_FREQUENCY = 100.  # 100 Hz
BANDFILTER_TYPE = 'hp'  # highpass
FILTER_SAMPLING_RATE = 1000. / DT  # 40 kHz

WINDOW_SIZE = 144 # time window for lfp
Y_WINDOW_SIZE = 1960. # y window size for lfp
PK_TR_IDX_IN_WINDOW = 20 # index in window to align first peak/trough of lfp
SPIKE_WINDOW = [-1.5, 4.] # spike time window for stochastic EAPs
START_IDX = 320 # for passive model
SOMA_INJECT_SCALING_FACTOR = 1085.  # 2.55

# SUMMARY STATISTICS LIST
SUMM_STATS_NAMES = np.array([
    'avg_mean', 'avg_std', 'avg_max_idx_x', 'avg_max_idx_y', 'avg_max_val', 'avg_min_idx_x', 'avg_min_idx_y', 'avg_min_val', # 8
    't_tr_mean', 't_tr_std', 't_tr_max_idx_x', 't_tr_max_idx_y', 't_tr_max_val', 't_tr_min_idx_x', 't_tr_min_idx_y', 't_tr_min_val', # 16
    't_pk_mean', 't_pk_std', 't_pk_max_idx_x', 't_pk_max_idx_y', 't_pk_max_val', 't_pk_min_idx_x', 't_pk_min_idx_y', 't_pk_min_val', # 24
    'stdev_mean', 'stdev_std', 'stdev_max_idx_x', 'stdev_max_idx_y', 'stdev_max_val', # 29
    'tr_mean', 'tr_std', 'tr_max_idx_x', 'tr_max_idx_y', 'tr_max_val', # 34
    'pk_mean', 'pk_std', 'pk_max_idx_x', 'pk_max_idx_y', 'pk_max_val', # 39
    't0', 't1', 't2', 't0_half_l_idx_y', 't0_half_r_idx_y', 't2_half_l_idx_y', 't2_half_r_idx_y', 't1_max_idx_y', 't1_min_idx_y', # 48
    'tr_lambda_l', 'tr_lambda_r', 'tr_slope_l', 'tr_slope_r', # 52
    'pk_lambda_l', 'pk_lambda_r', 'pk_slope_l', 'pk_slope_r', # 56
    'tr_l_w1', 'tr_l_w2', 'tr_l_y1', 'tr_r_w1', 'tr_r_w2', 'tr_r_y1', # 62
    'pk_l_w1', 'pk_l_w2', 'pk_l_y1', 'pk_r_w1', 'pk_r_w2', 'pk_r_y1', # 68
    'tr_l_avg_mag', 'tr_r_avg_mag', 'pk_l_avg_mag', 'pk_r_avg_mag', # 72
    'tr_l_t1', 'tr_l_t2', 'tr_r_t1', 'tr_r_t2', # 76
    'pk_l_t1', 'pk_l_t2', 'pk_r_t1', 'pk_r_t2', # 80
    'tr_l_tprop', 'tr_r_tprop', 'pk_l_tprop', 'pk_r_tprop', # 84
    'log_avg_mean', 'log_avg_std', 'log_stdev_mean', 'log_stdev_std', # 88
    'log_tr_mean', 'log_tr_std', 'log_pk_mean', 'log_pk_std', # 92
    'avg_mean_p1', 'avg_std_p1', 'avg_mean_p2', 'avg_std_p2', 'avg_mean_p3', 'avg_std_p3', 'avg_mean_p4', 'avg_std_p4', # 100
    'stdev_mean_p1', 'stdev_std_p1', 'stdevg_mean_p2', 'stdev_std_p2', 'stdev_mean_p3', 'stdev_std_p3', 'stdev_mean_p4', 'stdev_std_p4', # 108
    'log_avg_mean_p1', 'log_avg_std_p1', 'log_avg_mean_p2', 'log_avg_std_p2', 'log_avg_mean_p3', 'log_avg_std_p3', 'log_avg_mean_p4', 'log_avg_std_p4', # 116
    'log_stdev_mean_p1', 'log_stdev_std_p1', 'log_stdev_mean_p2', 'log_stdev_std_p2', 'log_stdev_mean_p3', 'log_stdev_std_p3', 'log_stdev_mean_p4', 'log_stdev_std_p4', # 124
])
# additional_stats 7
def additional_stats_7():
    n_parts_t, n_parts_y = 4, 4
    mag_scale = ['linear', 'log']
    stats_type = ['avg', 'stdev']
    mesh_parts = np.meshgrid(mag_scale, range(1, n_parts_t + 1), range(1, n_parts_y + 1), stats_type, indexing='ij')
    stats_grid = [log + '_' + st + f'_t{pt:d}_y{py:d}' for log, pt, py, st in zip(*map(np.ravel, mesh_parts))]
    return stats_grid
SUMM_STATS_NAMES = np.concatenate((SUMM_STATS_NAMES, additional_stats_7()))
