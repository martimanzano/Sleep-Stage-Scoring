# DIRECTORIES
data_path = "data/"
stage_file_substr = "_stage.txt"

# CHANNELS TO USE
eeg_channel_name = 'C3A2'
eeg2_channel_name = 'C4A1'
eog1_channel_name = 'Lefteye'
eog2_channel_name = 'RightEye'
emg_channel_name = 'EMG'
channels_to_use = ['EEG']#, 'EEG2']#, 'EOG1', 'EOG2', 'EMG']

# SIGNAL INFO & PROCESSING
sample_freq = 128
cutoff_EEG = 0.3
cutoff_EOG =  0.4
cutoff_EMG = 0.5
normalize_sum_factor = 0.5
downsampling_factor = 2
downsampling_factor_freq = 2
epoch_duration_s = 30
num_labels = 5

# LEARNING PARAMETERS
k_cross_valid = 'LOO' # 'LOO' for Leave One Out
randomize_patients = False
num_steps_GD = 2000
pretraining_steps = num_steps_GD / 10
SGD_batch_size = 512
GD_early_stopping_steps = num_steps_GD / 10
base_learning_rate = 0.01
dropout_keep_rate = 0.5
balance_classes = False
print_status_every = 20
pretrain = False
num_pre_post_SSL = 6
