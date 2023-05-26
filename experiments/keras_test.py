from keras.layers import Dense, Dropout, Activation, LSTM, GRU, SimpleRNN, Convolution1D, MaxPooling1D, Embedding, Flatten
from keras.models import Sequential
import config
import util as util
import numpy as np

from signal_tools import *

def run(signals, stages):
    # Convolution
    filter_length = 5
    nb_filter = 64
    pool_length = 4

    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 100
    nb_epoch = 50


    print('Loading data...')
    ### LANZAMIENTO DE LA SESION

    k_cross_validation = util.get_valid_k_cross_validation(len(signals),
                                                           config.k_cross_valid)

    for training_x, training_y, validation_x, validation_y \
            in util.k_fold_cross_validation(signals, stages, k_cross_validation):
        # SET UP DATA VARIABLES
        ## PREPARAR EPOCHS DE 30 SEGUNDOS. INICIALMENTE POR PACIENTES SEPARADAMENTE
        ## ESTRUCTURA DE ETIQUETAS EN 1-HOT ENCODING
        ## CONJUNTOS POR PACIENTES: FUSIONAR DATOS
        labels_size_training_list = [len(x) for x in training_y]

        epochs_training_x = divide_in_epochs(training_x, labels_size_training_list,
                                             config.sample_freq, config.epoch_duration_s)

        epochs_training_x, training_y = util.down_balance(epochs_training_x,
                                                        np.array(list(itertools.chain.from_iterable(training_y))))

        epochs_training_y = training_y

        labels_size_validation_list = [len(x) for x in validation_y]
        epochs_validation_x = divide_in_epochs(validation_x, labels_size_validation_list,
                                               config.sample_freq, config.epoch_duration_s)
        epochs_validation_y = np.array(list(itertools.chain.from_iterable(validation_y)))

        model = Sequential()
        # model.add(Dense(32,input_dim=(config.sample_freq * config.epoch_duration_s * util.get_number_of_channels(),
        #                 config.num_labels)))
        # model.add(Dense(128, input_shape=(960,)))
        model.add(Convolution1D(input_dim=960,
                                nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1,
                                input_length=np.shape(epochs_training_y)[0]))

        model.add(MaxPooling1D(pool_length=2))
        model.add(Flatten())
        model.add(Dense(config.num_labels))
        model.add(Dropout(0.25))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('Train...')
        model.fit(epochs_training_x[None,:,:], epochs_training_y, batch_size=batch_size, nb_epoch=nb_epoch,
                  validation_data=(epochs_validation_x, epochs_validation_y))
        score, acc = model.evaluate(epochs_validation_x, epochs_validation_y, batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)