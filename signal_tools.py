from scipy import signal, interpolate
import scipy.fftpack as sp
import numpy as np
import itertools
import config
import pywt
import util
import matplotlib.pyplot as plt


def notch_filter(freq_filt, sample_freq, original_signal):
    stop_Hz = np.array([freq_filt - 1, freq_filt + 1])

    b, a = signal.butter(2, stop_Hz / (sample_freq / 2.0), 'bandstop')
    return signal.filtfilt(b, a, original_signal)


def bandpass_filter(freq_filt, sample_freq, original_signal):
    #b, a = signal.butter(2, freq_filt * (2.0 / sample_freq), 'highpass')
    b, a = signal.butter(2, round(freq_filt * (2.0 / sample_freq), 4), 'highpass')
    #b, a = signal.butter(2, [freq_filt * (2.0 / sample_freq), 32 * (2.0 / sample_freq)], btype="band")
    temp = signal.filtfilt(b, a, original_signal)
    return temp
    # b, a = signal.butter(2, round(40 * (2.0 / sample_freq), 4), 'lowpass')
    # return signal.filtfilt(b, a, temp)


def downsample(original_signal, factor):
    return original_signal[::factor]
    #return signal.decimate(original_signal, q=factor)


def downsample_epochs_set(epochs_set, factor):
    return np.apply_along_axis(downsample, 1, epochs_set, factor)
    # np.array([downsample(x, config.downsampling_factor/2) for x in epochs_training_x])


def current_sample_freq_t():
    return config.sample_freq / config.downsampling_factor


def current_sample_freq_f():
    return config.sample_freq / config.downsampling_factor_freq


def cutoff(original_signal, cutoff_f):
    original_signal[np.where(original_signal < -cutoff_f)] = -cutoff_f
    original_signal[np.where(original_signal > cutoff_f)] = cutoff_f
    return original_signal


def normalize(original_signal, cutoff_f, normalize_sum_factor):
    original_signal = original_signal / (cutoff_f * 2) + normalize_sum_factor
    return original_signal


def discrete_wavelet_transform(original_signal):
    cA, cD = pywt.dwt(original_signal, 'db2')
    return cA, cD


def interp(ys, mul):
    # linear extrapolation for last (mul - 1) points
    ys = list(ys)
    ys.append(2*ys[-1] - ys[-2])
    # make interpolation function
    xs = np.arange(len(ys))
    fn = interpolate.interp1d(xs, ys, kind="linear")
    # call it on desired data points
    new_xs = np.arange(len(ys) - 1, step=1./mul)
    return fn(new_xs)


def convert_stage_labels(stages):
    temp_stages = np.copy(stages)

    stages[np.where(temp_stages == 0)] = 5  # WAKE
    stages[np.where(temp_stages == 1)] = 4  # REM
    stages[np.where(temp_stages == 2)] = 3  # STAGE 1
    stages[np.where(temp_stages == 3)] = 2  # STAGE 2
    stages[np.where(temp_stages == 4)] = 1  # STAGE 3 DEEP SLEEP
    stages[np.where(temp_stages == 5)] = 1  # STAGE 4 DEEP SLEEP
    stages[np.where(temp_stages == 6)] = 0  # ARTIFACT
    stages[np.where(temp_stages == 7)] = 0  # INDETERM
    stages[np.where(temp_stages == 8)] = 0  # INDETERM

    return stages


def divide_in_epochs(original_signal, labels_size_list, sample_freq, epoch_duration_s):
    # Llegan varios arrays gigantes (uno por cada paciente), cada uno tiene
    # todas las muestras recogidas para el periodo de extraccion de la senal
    # Esta funcion tiene que: coger cada array gigante y dividirlo en arrays
    # menores que seran los epochs
    epoch_size = sample_freq * epoch_duration_s * util.get_number_of_channels()
    #def get_array_of_epochs_size(signal_s, labels_size_list_, freq, epoch_duration):
    def get_array_of_epochs_size(labels_size_list_):
        size_ = 0
        #for single_signal_array_, labels_size_ in signal_s, labels_size_list_:
        for labels_size_ in labels_size_list_:
            #subarray_size_ = labels_size_ * epoch_size #len(single_signal_array_) / (freq * epoch_duration)
            size_ += labels_size_
        return size_

    size = get_array_of_epochs_size(labels_size_list)
    array_of_epochs = np.zeros(shape=(size, epoch_size))

    index_epochs = 0
    for single_signal_array, labels_size in itertools.izip(original_signal, labels_size_list):
        #subarray_size = int(ceil((len(single_signal_array) / sample_freq) / float(epoch_duration_s)))
        subarray_size = labels_size * epoch_size
        single_signal_array = single_signal_array[0:subarray_size]
        subarray_of_epochs = np.array_split(single_signal_array, labels_size)
        for epoch in subarray_of_epochs:
            array_of_epochs[index_epochs] = epoch
            index_epochs += 1
    return array_of_epochs


def join_stages_hot_enc(stages, join=True):
    if join:
        joined_stages = np.array(list(itertools.chain.from_iterable(stages)))
    else:
        joined_stages = stages

    joined_stages = np.array([x - 1 for x in joined_stages])
    labels_hot_enc = (np.arange(config.num_labels) == joined_stages[:, None]).astype(np.float32)
    return labels_hot_enc


def preprocess_epochs_set(epochs_set):
    for i, raw_epoch in enumerate(epochs_set):
        # do preprocessing stuff
        # epochs_set[i] = notch_filter(50, config.sample_freq, raw_epoch)
        # epochs_set[i] = bandpass_filter(0.3, config.sample_freq, raw_epoch)
        epochs_set[i] = cutoff(raw_epoch, config.cutoff_EEG)
        # epochs_set[i] = normalize(raw_epoch, config.cutoff_EEG, config.normalize_sum_factor)
    return epochs_set


def get_freq_domain_epochs(time_epochs_set):
    size_freq_epoch = (current_sample_freq_f() / 2) * config.epoch_duration_s
    freq_epochs_set = np.zeros(shape=(time_epochs_set.shape[0], size_freq_epoch))
    for i, time_epoch in enumerate(time_epochs_set):
        # p = np.array(20*np.log10(np.abs(np.fft.rfft(time_epoch))))
        # p = np.abs(np.fft.rfft(time_epoch))
        # p = abs(sp.fft(time_epoch))
        # p = p*2/len(p)
        hann = np.hanning(len(time_epoch)) * time_epoch
        phann = abs(sp.fft(hann))
        phann = phann*2*2/len(phann)
        # f = np.linspace(0, config.sample_freq/2, len(p))
        freq_epochs_set[i] = phann[:len(phann)/2]
        #plt.plot(phann[:len(phann)/2])
        #plt.show()
        #plt.figure(1)
        #plt.plot(time_epoch)
        #plt.figure(2)
        # plt.plot(f, p)
        #plt.show()
    return freq_epochs_set





