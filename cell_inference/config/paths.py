import os
import platform

ROOT_DIR = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
LIBRARY = 'nrnmech.dll' if platform.system() == 'Windows' else 'libnrnmech.so'
COMPILED_LIBRARY = os.path.join(ROOT_DIR, 'resources/compiled/x86_64/.libs', LIBRARY)
MECHANISMS = os.path.join(ROOT_DIR, 'resources/compiled/mechanisms')
GEO_STANDARD = os.path.join(ROOT_DIR, 'resources/geom_standard.csv')
ELECTRODES = os.path.join(ROOT_DIR, 'resources/Electrode2D.h5')

SIMULATED_DATA_FILE = os.path.join(ROOT_DIR, 'resources/active_groundtruth.h5')
MORPHOLOGY_DATA_FILE = os.path.join(ROOT_DIR, 'resources/detailed_groundtruth.h5')  # Allen Morphology Cell Data
INVIVO_DATA_FILE = os.path.join(ROOT_DIR, 'resources/cell360LFP.h5')
INVIVO2D_DATA_FILE = os.path.join(ROOT_DIR, 'resources/cell360LFP2D.h5')

INFERENCE_RESULTS_ROOT = os.path.join(ROOT_DIR, 'results')

PASSIVE_INFERENCE_SAVE_TRACES = os.path.join(INFERENCE_RESULTS_ROOT,
                                             'result_pdfs/PassiveCellResultsCNN_Traces.pdf')
PASSIVE_INFERENCE_SAVE_HEATMAPS = os.path.join(INFERENCE_RESULTS_ROOT,
                                               'result_pdfs/PassiveCellResultsCNN_HTmap.pdf')
PASSIVE_INFERENCE_SAVE_KDE = os.path.join(INFERENCE_RESULTS_ROOT,
                                          'result_pdfs/PassiveCellResultsCNN_KDE.pdf')

PASSIVE_INFERENCE_RESULTS_DATA = os.path.join(INFERENCE_RESULTS_ROOT,
                                              'result_data/PassiveCellResultsCNN_Data.h5')
PASSIVE_INFERENCE_RESULTS_MATLAB_DATA = os.path.join(INFERENCE_RESULTS_ROOT,
                                                     'result_data/PassiveCellResultsCNN_MATData.mat')
PASSIVE_INFERENCE_RESULTS_X0_MATLAB_DATA = os.path.join(INFERENCE_RESULTS_ROOT,
                                                        'result_data/PassiveCellResultsCNN_x0.mat')

POSTERIOR_SAVE = os.path.join(ROOT_DIR, 'results/Posteriors/')

INFERENCER_SAVE = os.path.join(ROOT_DIR, 'results/Inferencers/')

IMAGE_SAVE = os.path.join(ROOT_DIR, 'results/ResultJPGs/')

ACTIVE_INFERENCE_SAVE_TRACES = os.path.join(INFERENCE_RESULTS_ROOT,
                                            'result_pdfs/ActiveCellResultsCNN_Traces.pdf')
ACTIVE_INFERENCE_SAVE_HEATMAPS = os.path.join(INFERENCE_RESULTS_ROOT,
                                              'result_pdfs/ActiveCellResultsCNN_HTmap.pdf')
ACTIVE_INFERENCE_SAVE_KDE = os.path.join(INFERENCE_RESULTS_ROOT,
                                         'result_pdfs/ActiveCellResultsCNN_KDE.pdf')

ACTIVE_INFERENCE_RESULTS_DATA = os.path.join(INFERENCE_RESULTS_ROOT,
                                             'result_data/ActiveCellResultsCNN_Data.h5')
ACTIVE_INFERENCE_RESULTS_MATLAB_DATA = os.path.join(INFERENCE_RESULTS_ROOT,
                                                    'result_data/ActiveCellResultsCNN_MATData.mat')
ACTIVE_INFERENCE_RESULTS_X0_MATLAB_DATA = os.path.join(INFERENCE_RESULTS_ROOT,
                                                       'result_data/ActiveCellResultsCNN_x0.mat')