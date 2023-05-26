from signal_tools import *
import config
import util as util
import numpy as np
import pyedflib
import glob
import os
import time

from experiments import SSL_Convolutional_2


def main():
    print "Signal pre-processing..."
    print "Using channels: " + str(config.channels_to_use)
    np.set_printoptions(precision=4)
    files_rec = sorted(glob.glob(config.data_path + "*.rec"))
    files_edf = sorted(glob.glob(config.data_path + "*.edf"))
    files_signals = files_rec + files_edf
    print files_signals


    files_stages = [os.path.splitext(x)[0] + config.stage_file_substr for x in files_signals]
    #print files_stages

    valid_idx_files = util.get_valid_pacients(files_signals, files_stages)
    #print valid_idx_files

    read_signals = np.empty(len(valid_idx_files), dtype=object)
    eeg_signal = np.empty(len(valid_idx_files), dtype=object)
    eog_signal = np.empty(len(valid_idx_files), dtype=object)
    emg_signal = np.empty(len(valid_idx_files), dtype=object)
    stages = np.empty(len(valid_idx_files), dtype=object)
    simul_time = np.zeros(len(valid_idx_files))

    print valid_idx_files
    for i,valid_idx in enumerate(valid_idx_files):
        print i, valid_idx, files_signals[valid_idx],

        with open(files_stages[valid_idx]) as stage_file:
            stages[i] = np.array([int(x) for x in stage_file.read().splitlines()])
            #print(stages[i][0:10])
            #print sum(1 for j in stages[i] if j == 7 or j == 6)
            stages[i] = convert_stage_labels(stages[i])
            #print(stages[i][0:10])
        # valid_pacient._close()
        # del valid_pacient

        #print "NAN OR INF INDEXES",
        # assert sum(1 for j in read_signals[i] if j is np.nan or j is np.inf) == 0
        #print " ",

    ## PREPARAR DATOS
    # config.sample_freq /= config.downsampling_factor
    print "Input size before preprocessing: " + \
           str(config.sample_freq * config.epoch_duration_s * util.get_number_of_channels())
    #print np.shape(read_signals)


    ## LANZAR EXPERIMENTO
    with open("signals", "r") as signals_file:
        # np.save(signals_file, read_signals)
        read_signals = np.load(signals_file)


    SSL_Convolutional_2.run(read_signals, stages)
    # SGD_perceptron.run(read_signals, stages)
    # keras_test.run(read_signals, stages)
    # ML_GD_perceptron.run(matlab_signals, matlab_labels)
    # autoencoder.run(eeg_signal, stages)

    ## FIN

if __name__ == "__main__":
    main()