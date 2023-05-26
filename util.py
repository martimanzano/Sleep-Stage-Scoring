import pyedflib
import config
import os
import numpy as np
from sklearn import metrics
import itertools
import glob

def get_valid_pacients(files_signals, files_stages):
    valid_idx_files = list()
    for curr_idx, fname in enumerate(files_signals):  # Aqui toca leer las senales y las anotaciones
        with pyedflib.EdfReader(fname) as pacient:
            signal_labels = pacient.getSignalLabels()
            eeg_ok = ('EEG' in config.channels_to_use and signal_labels.index(config.eeg_channel_name) != -1) or not 'EEG' in config.channels_to_use
            eog1_ok = ('EOG1' in config.channels_to_use and signal_labels.index(config.eog1_channel_name) != -1) or not 'EOG1' in config.channels_to_use
            eog2_ok = ('EOG2' in config.channels_to_use and signal_labels.index(config.eog2_channel_name) != -1) or not 'EOG2' in config.channels_to_use
            emg_ok = ('EMG' in config.channels_to_use and signal_labels.index(config.emg_channel_name) != -1) or not 'EMG' in config.channels_to_use

            if eeg_ok and eog1_ok and eog2_ok and emg_ok and os.path.isfile(files_stages[curr_idx]):
                valid_idx_files.append(curr_idx)
    return valid_idx_files


def get_valid_k_cross_validation(valid_dataset_size, k_param):
    if type(k_param) is str and k_param.upper() == "LOO":
        return valid_dataset_size
    return k_param


def k_fold_cross_validation(X, Y, K, randomise = False):
    """
	Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
    print str(K) + "-Cross Validation. Training on: " + str((K-1)*len(X)/K) + " patients. Testing on " + str(len(X)/K)
    # if randomise: from random import shuffle; X=list(X); shuffle(X)
    if randomise: from sklearn import utils; X, Y = utils.shuffle(X, Y)
    for k in xrange(K):
        training_x = [x for i, x in enumerate(X) if i % K != k]
        training_y = [y for i, y in enumerate(Y) if i % K != k]

        validation_x = [x for i, x in enumerate(X) if i % K == k]
        validation_y = [y for i, y in enumerate(Y) if i % K == k]

        yield training_x, training_y, validation_x, validation_y


def down_balance(X, Y):
    min_samples_per_class = np.min(np.bincount(Y)[1:])
    balanced_X = np.zeros([min_samples_per_class * (config.num_labels), np.shape(X)[1]], X.dtype)
    balanced_Y = np.zeros(min_samples_per_class * (config.num_labels), Y.dtype)

    for i in range(1, config.num_labels + 1):
        indices_class = np.random.choice(np.where(Y == i)[0], min_samples_per_class)
        balanced_X[(i - 1) * min_samples_per_class:i * min_samples_per_class, :] = X[indices_class, :]
        # for idx, j in enumerate(indices_class):
        #     balanced_X[(i - 1) * min_samples_per_class + idx, :] = X[j]
        balanced_Y[(i - 1) * min_samples_per_class:i * min_samples_per_class] = Y[indices_class]
    return balanced_X, balanced_Y


def get_number_of_channels():
    return len(config.channels_to_use)


def get_channel_code(channel):
    if channel == 'EEG':
        chn_name = config.eeg_channel_name
    elif channel == 'EEG2':
        chn_name = config.eeg2_channel_name
    elif channel == 'EOG1':
        chn_name = config.eog1_channel_name
    elif channel == 'EOG2':
        chn_name = config.eog2_channel_name
    elif channel == 'EMG':
        chn_name = config.emg_channel_name
    else:
        raise NameError('Unknown channel name')
    return chn_name


def get_f1_score(y_labels, y_pred):
    return metrics.f1_score(np.array(list(itertools.chain.from_iterable(y_labels))), y_pred + 1, average='macro')

def get_confusion_matrix(y_labels, y_pred):
    return metrics.confusion_matrix(np.array(list(itertools.chain.from_iterable(y_labels))), y_pred + 1)


def print_main_metrics(train_acc, val_acc, val_f1):
    print(" Whole Training Acc: {:.5f}".format(train_acc))
    print("Validation Accuracy: {:.5f}".format(val_acc))
    print("Validation F1-Score: {:.5f}".format(val_f1))

def store_patient_ssl_data(index, true_labels, predicted_labels):
    dirname = "SSLdata/SSLpatient_" + str(index)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    true_labels_filename = dirname + "/" + "true_labels" + str(index) + ".txt"
    with open(true_labels_filename, "w") as true_labels_file:
        np.savetxt(true_labels_file, true_labels, fmt='%i', delimiter='\n')

    predicted_labels_filename = dirname + "/" + "predicted_labels" + str(index) + ".txt"
    with open(predicted_labels_filename, "w") as predicted_labels_file:
        np.savetxt(predicted_labels_file, predicted_labels, fmt='%i', delimiter='\n')


def get_ssl_data_cv_iteration(index):
    dirname = "SSLdata/"
    folders_SSL = sorted(glob.glob(dirname + "*/"), key=get_nkey)
    ssl_data = np.empty(len(folders_SSL) - 1, dtype=object)

    ssl_data_aux_idx = 0
    for idx, folder in enumerate(folders_SSL):
        if (idx + 1) != index:
            # CARGAR ETIQUETAS PACIENTE ACTUAL
            file_i = str(idx + 1)
            current_patient_filename = dirname + "SSLpatient_" + file_i + "/predicted_labels" + file_i + ".txt"
            with open(current_patient_filename, "r") as current_patient:
                current_patient_data = np.loadtxt(current_patient, dtype=int, delimiter='\n')
                ssl_data[ssl_data_aux_idx] = current_patient_data
            ssl_data_aux_idx += 1
    return np.array(list(itertools.chain.from_iterable(ssl_data)))
    #return ssl_data

def get_ssl_data_complete():
    dirname = "SSLdata/"
    folders_SSL = sorted(glob.glob(dirname + "*/"), key=get_nkey)
    ssl_data = np.empty(len(folders_SSL), dtype=object)

    for idx, folder in enumerate(folders_SSL):
        # CARGAR ETIQUETAS PACIENTE ACTUAL
        file_i = str(idx + 1)
        current_patient_filename = dirname + "SSLpatient_" + file_i + "/predicted_labels" + file_i + ".txt"
        with open(current_patient_filename, "r") as current_patient:
            current_patient_data = np.loadtxt(current_patient, dtype=int, delimiter='\n')
            ssl_data[idx] = current_patient_data
    return ssl_data

def get_nkey(filename):
    return int(filename.split("_")[1].split("/")[0])


def build_ssl_stages_data(predictions):
    ssl_single_structure_size = (config.num_pre_post_SSL * 2) + 1
    # ssl_stages_data = np.empty(len(predictions), dtype=np.zeros(ssl_single_structure_size, dtype=np.int))
    ssl_stages_data = np.empty(len(predictions), dtype=object)

    ssl_stage = np.zeros(ssl_single_structure_size, dtype=np.int)
    for i, prediction in enumerate(predictions):
        ssl_stage[0:config.num_pre_post_SSL - 1] = predictions[i - config.num_pre_post_SSL:i - 1] # ESTO ESTA MAL. HAY QUE INCORPORAR LOS CASOS DE LOS EXTREMOS
        ssl_stage[config.num_pre_post_SSL] = prediction
        ssl_stage[config.num_pre_post_SSL + 1:ssl_single_structure_size - 1] = predictions[i + 1: i + config.num_pre_post_SSL]

        ssl_stages_data[i] = ssl_stage
