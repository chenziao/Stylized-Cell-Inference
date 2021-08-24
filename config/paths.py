import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from config.params import OSVERSION

ROOT_DIR = "/".join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
LIBRARY = "libnrnmech.so" if OSVERSION == 'Linux' else 'nrnmech.dll'
COMPILED_LIBRARY = os.path.join(ROOT_DIR, 'compiled/x86_64/.libs', LIBRARY)
MECHANISMS = os.path.join(ROOT_DIR, 'compiled/mechanisms')
GEO_STANDARD = os.path.join(ROOT_DIR, 'config/geom_standard.csv')

SIMULATED_DATA_FILE = os.path.join(ROOT_DIR, 'data/active_groundtruth.h5')
MORPHOLOGY_DATA_FILE = os.path.join(ROOT_DIR, 'data/detailed_groundtruth.h5') #Allen Morphology Cell Data
INVIVO_DATA_FILE = os.path.join(ROOT_DIR, 'data/cell360LFP.h5')

INFERENCE_RESULTS_ROOT = os.path.join(ROOT_DIR, 'results')

PASSIVE_INFERENCE_SAVE_TRACES = os.path.join(INFERENCE_RESULTS_ROOT, 
                                            'ResultPDFs/PassiveCellResultsCNN_Traces.pdf')
PASSIVE_INFERENCE_SAVE_HEATMAPS = os.path.join(INFERENCE_RESULTS_ROOT,
                                            'ResultPDFs/PassiveCellResultsCNN_HTmap.pdf')
PASSIVE_INFERENCE_SAVE_KDE = os.path.join(INFERENCE_RESULTS_ROOT, 
                                            'ResultPDFs/PassiveCellResultsCNN_KDE.pdf')

PASSIVE_INFERENCE_RESULTS_DATA = os.path.join(INFERENCE_RESULTS_ROOT, 
                                            'ResultData/PassiveCellResultsCNN_Data.h5')
PASSIVE_INFERENCE_RESULTS_MATLAB_DATA = os.path.join(INFERENCE_RESULTS_ROOT, 
                                            'ResultData/PassiveCellResultsCNN_MATData.mat')
PASSIVE_INFERENCE_RESULTS_X0_MATLAB_DATA = os.path.join(INFERENCE_RESULTS_ROOT, 'ResultData/PassiveCellResultsCNN_x0.mat')

POSTERIOR_SAVE = os.path.join(ROOT_DIR, 'results/Posteriors/')

INFERENCER_SAVE = os.path.join(ROOT_DIR, 'results/Inferencers/')

IMAGE_SAVE = os.path.join(ROOT_DIR, 'results/ResultJPGs/')


ACTIVE_INFERENCE_SAVE_TRACES = os.path.join(INFERENCE_RESULTS_ROOT, 
                                            'ResultPDFs/ActiveCellResultsCNN_Traces.pdf')
ACTIVE_INFERENCE_SAVE_HEATMAPS = os.path.join(INFERENCE_RESULTS_ROOT,
                                            'ResultPDFs/ActiveCellResultsCNN_HTmap.pdf')
ACTIVE_INFERENCE_SAVE_KDE = os.path.join(INFERENCE_RESULTS_ROOT, 
                                            'ResultPDFs/ActiveCellResultsCNN_KDE.pdf')

ACTIVE_INFERENCE_RESULTS_DATA = os.path.join(INFERENCE_RESULTS_ROOT, 
                                            'ResultData/ActiveCellResultsCNN_Data.h5')
ACTIVE_INFERENCE_RESULTS_MATLAB_DATA = os.path.join(INFERENCE_RESULTS_ROOT, 
                                            'ResultData/ActiveCellResultsCNN_MATData.mat')
ACTIVE_INFERENCE_RESULTS_X0_MATLAB_DATA = os.path.join(INFERENCE_RESULTS_ROOT, 'ResultData/ActiveCellResultsCNN_x0.mat')