import os, sys
sys.path.append(os.path.split(sys.path[0])[0])

import glob
import os
import pandas as pd
import numpy as np
import collections

import config.paths as paths

def build_input_csv():
    # get all the csv files in that directory (assuming they have the extension .csv)
    csvfiles = glob.glob(os.path.join(paths.CSV_TEMP_IN_FILES, '*.csv'))

    # loop through the files and read them in with pandas
    df = {}  # a list to hold all the individual pandas DataFrames
    for csvfile in csvfiles:
        csv = open(csvfile)
        ds = pd.read_csv(csvfile, header=None)
        n = (csv.name.split('_')[-1]).split('.')[0]
        df[n] = ds

    od = collections.OrderedDict(sorted(df.items()))

    temp = []
    for k, v in od.items():
        temp.append(v.to_numpy()[int(k)::len(df),:])

    con_arr = np.concatenate(temp)

    np.savetxt(paths.CSV_SIM_IN_FILE, con_arr, delimiter=",")


def build_lfp_csv():
    csvfiles = glob.glob(os.path.join(paths.CSV_TEMP_LFP_FILES, '*.csv'))

    # loop through the files and read them in with pandas
    df = {}  # a list to hold all the individual pandas DataFrames
    for csvfile in csvfiles:
        csv = open(csvfile)
        ds = pd.read_csv(csvfile, header=None)
        n = (csv.name.split('_')[-1]).split('.')[0]
        df[n] = ds

    od = collections.OrderedDict(sorted(df.items()))

    temp = []
    for k, v in od.items():
        temp.append(v)

    con_arr = np.concatenate(temp)

    np.savetxt(paths.CSV_SIM_LFP_FILE, con_arr, delimiter=",")