import tensorflow as tf
import config
import util as util
import numpy as np
from signal_tools import *
from sklearn import metrics


def run(signals, stages):
### CREACION DE GRAFO DE COMPUTO
    graph = tf.Graph()
    with graph.as_default():
        ## NETWORK PARAMETERS
        n_hidden_1 = 200
        n_hidden_2 = 200
        n_input = config.sample_freq * config.epoch_duration_s * 4
        n_classes = config.num_labels

        ## INPUT DATA. RECEIVED FROM THE SESSION
        tf_data_x = tf.placeholder(tf.float32, [None, n_input])   #shape=(config.SGD_batch_size, n_input))
        tf_data_y = tf.placeholder(tf.float32, [None, n_classes]) #shape=(config.SGD_batch_size, n_classes))

        #tf_valid_dataset = tf.placeholder(tf.float32, [None, n_input])
        #tf_valid_labels = tf.placeholder(tf.float32, [None, n_classes])

        ## MODEL SET UP
        def multilayer_perceptron(x_data, weights, biases):
            # HIDDEN LAYER 1 WITH RELU ACTIVATION
            layer_1 = tf.add(tf.matmul(x_data, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # HIDDEN LAYER 2 WITH RELU ACTIVATION
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            # OUTPUT LAYER WITH LINEAL ACIVATION
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

        # WEIGHTS AND BIASES SET UP
        weights_str = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases_str = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # BUILD MODEL
        pred = multilayer_perceptron(tf_data_x, weights_str, biases_str)

        # LOSS AND OPTIMIZER SET UP
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, tf_data_y))
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # PREDICTIONS
        predictions = tf.arg_max(pred, 1)
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(tf_data_y, 1))


    ### LANZAMIENTO DE LA SESION
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Gradient Descent MultiLayer-Perceptron: Initializated")
        k_cross_validation = util.get_valid_k_cross_validation(len(signals),
                                                               config.k_cross_valid)
        cv_train_acc = 0
        cv_valid_acc = 0
        cv_f1_score = 0
        k_index = 1
        for training_x, training_y, validation_x, validation_y \
                in util.k_fold_cross_validation(signals, stages, k_cross_validation):
            l = np.inf
            ## PREPARAR VALIDACION CRUZADA
            labels_size_training_list = [len(x) for x in training_y]

            #epochs_training_x = divide_in_epochs(training_x, labels_size_training_list,
                                                 #config.sample_freq, config.epoch_duration_s)
            epochs_training_x = np.array(list(itertools.chain.from_iterable(training_x)))

            # epochs_training_x, training_y = util.down_balance(np.array(list(itertools.chain.from_iterable(epochs_training_x))),
            #                                                   np.array([int(i) for i in np.array(list(itertools.chain.from_iterable(np.array(
            #                                                       list(itertools.chain.from_iterable(training_y))))))]))

            epochs_training_y = join_stages_hot_enc(list(itertools.chain.from_iterable(training_y)), join=True)

            feed_dict_train = {tf_data_x: epochs_training_x, tf_data_y: epochs_training_y}
            labels_size_validation_list = [len(x) for x in validation_y]
            # epochs_validation_x = divide_in_epochs(validation_x, labels_size_validation_list,
            #                                        config.sample_freq, config.epoch_duration_s)
            epochs_validation_x = np.array(list(itertools.chain.from_iterable(validation_x)))

            epochs_validation_y = join_stages_hot_enc(np.array([int(i) for i in np.array(list(itertools.chain.from_iterable(validation_y)))]), join=False)
            feed_dict_test = {tf_data_x: epochs_validation_x, tf_data_y: epochs_validation_y}
            for step in range(config.num_steps_GD):
                ## PREPARAR EPOCHS DE 30 SEGUNDOS. INICIALMENTE POR PACIENTES SEPARADAMENTE
                ## ESTRUCTURA DE ETIQUETAS EN 1-HOT ENCODING
                ## CONJUNTOS POR PACIENTES: FUSIONAR DATOS
                ## TODO: BALANCEO DE CLASES

                # OPTIMIZAR MODELO PARA DATOS DE TRAIN
                _, l, correct_train_predictions = session.run(
                    [optimizer, loss, correct_predictions], feed_dict=feed_dict_train)
                if step % 10 == 0:
                    # CALCULAR MEDIDAS DE CALIDAD PARA TRAINING Y TEST DE LA C.V
                    test_predictions, correct_test_predictions = session.run([predictions, correct_predictions],
                                                                         feed_dict=feed_dict_test)
                    step_f1_score = metrics.f1_score(validation_y[0], test_predictions, average='macro')
                    step_train_acc = tf.reduce_mean(tf.cast(correct_train_predictions, "float"))
                    step_valid_acc = tf.reduce_mean(tf.cast(correct_test_predictions, "float"))
                    print("----------GD STEP %i of %i FINISHED----------" % (step + 1, config.num_steps_GD))
                    print("Loss: {:.5f}".format(l))
                    print("Training accuracy: {:.5f}".format(step_train_acc.eval()))
                    print("Validation accuracy: {:.5f}".format(step_valid_acc.eval()))
                    print("Validation F1-Score: {:.5f}".format(step_f1_score))


            # CALCULAR MEDIDAS DE CALIDAD PARA TRAINING Y TEST DE LA C.V
            test_predictions, correct_test_predictions = session.run([predictions, correct_predictions],
                                                                     feed_dict=feed_dict_test)

            partial_cv_f1_score = metrics.f1_score(validation_y[0], test_predictions, average='macro')
            partial_cv_valid_acc = tf.reduce_mean(tf.cast(correct_test_predictions, "float"))
            cv_valid_acc += partial_cv_valid_acc
            cv_f1_score += partial_cv_f1_score
            k_index += 1

            print("////////// CROSS.VAL %i of %i FINISHED //////////" % (k_index, k_cross_validation))
            print("Loss: {:.5f}".format(l))
            print("Validation accuracy: {:.5f}".format(partial_cv_valid_acc.eval()))
            print("Validation F1-Score: {:.5f}".format(partial_cv_f1_score))


        cv_valid_acc /= k_cross_validation
        cv_f1_score /= k_cross_validation
        print("////////// FINISHED //////////")
        print("Validation average accuracy: {:.5f}".format(cv_valid_acc.eval()))
        print("Validation average F1-Score: {:.5f}".format(cv_f1_score))