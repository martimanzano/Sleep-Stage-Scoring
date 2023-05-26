import tensorflow as tf
import tflearn as tflearn
import config
import util as util
import numpy as np
from best_step import BestStep
from signal_tools import *
from sklearn import metrics
from sklearn.utils import shuffle

def run(signals, stages):
### CREACION DE GRAFO DE COMPUTO
    graph = tf.Graph()
    with graph.as_default():
        ## NETWORK PARAMETERS
        n_in_width_1 = 32
        n_in_width_2 = 32
        n_out_depth_1 = 10
        n_out_depth_2 = 20
        n_hidden_fc = 500
        pool_factor_1 = 2
        pool_factor_2 = 2
        stride_conv_1 = 2
        stride_conv_2 = 2
        n_input = config.sample_freq * config.epoch_duration_s * util.get_number_of_channels()
        n_classes = config.num_labels

        ## INPUT DATA. RECEIVED FROM THE SESSION
        tf_data_x = tf.placeholder(tf.float32, [None, n_input])   #shape=(config.SGD_batch_size, n_input))
        tf_data_y = tf.placeholder(tf.float32, [None, n_classes]) #shape=(config.SGD_batch_size, n_classes))
        keep_prob = tf.placeholder(tf.float32)


        def conv1d(x, W, b, strides=1):
            x = tf.nn.conv1d(x, W, stride=strides, padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

        def maxpool2d(x, k=2):
            # return tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1, 1, k, 1], padding='SAME')
            return tflearn.layers.conv.max_pool_1d(x, k, strides=k, padding='VALID')

        ## MODEL SET UP
        def convolutional_net(x_data, weights, biases, dropout_prob):
            # Convolution layer 1
            print(x_data.get_shape())
            x_data = tf.reshape(x_data, [-1, n_input, 1])
            print(x_data.get_shape())
            conv1 = conv1d(x_data, weights['c1'], biases['bc1'], strides=2)
            print(conv1.get_shape())
            # Max Pooling 1
            conv1 = maxpool2d(conv1, pool_factor_1)
            print(conv1.get_shape())

            # Convolution layer 2
            conv2 = conv1d(conv1, weights['c2'], biases['bc2'], strides=2)
            print(conv2.get_shape())
            # Max Pooling 2
            conv2 = maxpool2d(conv2, pool_factor_2)
            print(conv2.get_shape())

            # Fully connected layer
            fc1 = tf.reshape(conv2, [-1, weights_str['d1'].get_shape().as_list()[0]])
            print(fc1.get_shape())
            fc1 = tf.add(tf.matmul(fc1, weights['d1']), biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            print(fc1.get_shape())
            # Apply Dropout regularization
            fc1 = tf.nn.dropout(fc1, dropout_prob)

            # Output -- Class prediction
            out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

            return out


        # WEIGHTS AND BIASES SET UP
        weights_str = {         # 4D: [height, width, in_ch, out_ch]
            'c1': tf.Variable(tf.random_normal([n_in_width_1, 1, n_out_depth_1])),
            'c2': tf.Variable(tf.random_normal([n_in_width_2, n_out_depth_1, n_out_depth_2])),
            # 'd1': tf.Variable(tf.random_normal([n_out_depth_2 * (n_input / (pool_factor_1 * pool_factor_2)),
            #                                     n_hidden_fc])),
            'd1': tf.Variable(tf.random_normal([n_out_depth_2 *
                                                (n_input / (pool_factor_1 * pool_factor_2 * stride_conv_1 * stride_conv_2)),
                                                                                    n_hidden_fc])),
            'out': tf.Variable(tf.random_normal([n_hidden_fc, n_classes]))
        }
        biases_str = {
            'bc1': tf.Variable(tf.random_normal([n_out_depth_1])),
            'bc2': tf.Variable(tf.random_normal([n_out_depth_2])),
            'bd1': tf.Variable(tf.random_normal([n_hidden_fc])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # BUILD MODEL
        pred = convolutional_net(tf_data_x, weights_str, biases_str, keep_prob)

        # LOSS AND OPTIMIZER SET UP
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, tf_data_y))
        # loss = tf.reduce_mean(tf.pow(tf_data_y - pred, 2))

        global_step = tf.Variable(0, trainable=False)
        # optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
        optimizer = tf.train.AdamOptimizer(config.base_learning_rate).minimize(loss, global_step=global_step)
        # optimizer = tf.train.RMSPropOptimizer(config.base_learning_rate).minimize(loss)

        # PREDICTIONS
        predictions = tf.arg_max(pred, 1)
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(tf_data_y, 1))


    ### LANZAMIENTO DE LA SESION
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Deep Convolutional Network: Initialized")
        k_cross_validation = util.get_valid_k_cross_validation(len(signals),
                                                               config.k_cross_valid)
        cv_train_acc = 0
        cv_valid_acc = 0
        cv_f1_score = 0
        k_index = 1
        for training_x, training_y, validation_x, validation_y \
                in util.k_fold_cross_validation(signals, stages, k_cross_validation):
            # SET UP CONTROL VARIABLES
            tf.initialize_all_variables().run()
            l = np.inf
            best_val_step = BestStep(0, 0, 0, 0)
            last_step_val_acc = 0
            early_stop_remaining_steps = config.GD_early_stopping_steps

            # SET UP DATA VARIABLES
            ## PREPARAR EPOCHS DE 30 SEGUNDOS. INICIALMENTE POR PACIENTES SEPARADAMENTE
            ## ESTRUCTURA DE ETIQUETAS EN 1-HOT ENCODING
            ## CONJUNTOS POR PACIENTES: FUSIONAR DATOS
            labels_size_training_list = [len(x) for x in training_y]

            epochs_training_x = divide_in_epochs(training_x, labels_size_training_list,
                                                 config.sample_freq, config.epoch_duration_s)
            print("Training epoches: " + str(len(epochs_training_x)))

            epochs_training_x, training_y = util.down_balance(epochs_training_x,
                                                            np.array(list(itertools.chain.from_iterable(training_y))))

            epochs_training_y = join_stages_hot_enc(training_y, join=False)
            ## SHUFFLE EPOCHS AFTER BALANCING ##
            epochs_training_x, epochs_training_y = shuffle(epochs_training_x, epochs_training_y)

            # feed_dict_train = {tf_data_x: epochs_training_x, tf_data_y: epochs_training_y, keep_prob: 0.75}
            labels_size_validation_list = [len(x) for x in validation_y]
            epochs_validation_x = divide_in_epochs(validation_x, labels_size_validation_list,
                                                   config.sample_freq, config.epoch_duration_s)
            epochs_validation_y = join_stages_hot_enc(validation_y)
            feed_dict_test = {tf_data_x: epochs_validation_x, tf_data_y: epochs_validation_y, keep_prob: 1}
            for step in range(config.num_steps_GD):
                # PREPARAR MINI BATCH A PARTIR DE LOS EPOCHS DE ENTRENAMIENTO
                ## ANULAR MINI BATCH ##
                # config.SGD_batch_size = epochs_training_y.shape[0]
                # offset = 0
                ##
                offset = (step * config.SGD_batch_size) % (epochs_training_y.shape[0] - config.SGD_batch_size)
                batch_data_x = epochs_training_x[offset:(offset + config.SGD_batch_size), :]
                batch_data_y = epochs_training_y[offset:(offset + config.SGD_batch_size), :]
                feed_dict_train = {tf_data_x: batch_data_x, tf_data_y: batch_data_y, keep_prob: 0.7}

                # GD: OPTIMIZAR MODELO PARA DATOS DE TRAIN
                _, l, correct_train_predictions = session.run(
                    [optimizer, loss, correct_predictions], feed_dict=feed_dict_train)

                # CALCULAR MEDIDAS DE CALIDAD PARA TRAINING Y TEST DE LA C.V
                test_predictions, correct_test_predictions = session.run([predictions, correct_predictions],
                                                                         feed_dict=feed_dict_test)

                step_valid_acc = np.mean(correct_test_predictions)
                if step_valid_acc > best_val_step.val_acc:
                    best_val_step.set_conv(step, tf.reduce_mean(tf.cast(correct_train_predictions, "float")).eval(),
                                      step_valid_acc, util.get_f1_score(validation_y, test_predictions),
                                      l, weights_str, biases_str)
                elif step_valid_acc < last_step_val_acc:
                    early_stop_remaining_steps -= 1
                    if early_stop_remaining_steps == 0:
                        print("EARLY-STOPPING OPTIMIZATION DUE TO " + str(config.GD_early_stopping_steps)
                              + " STEPS WITH VALIDATION ACC. GETTING WORSE...")
                        break
                elif step_valid_acc >= last_step_val_acc:
                    early_stop_remaining_steps = config.GD_early_stopping_steps

                if step % 10 == 0: # PINTADO DE DATOS DE LA OPTIMIZACION
                    # COMO OPTIMIZAMOS POR MINI BATCH, PRIMERO RECOGER PRECISION DE TRAIN PARA EL CONJUNTO ENTERO
                    feed_dict_whole_train = {tf_data_x: epochs_training_x, tf_data_y: epochs_training_y, keep_prob: 1}
                    whole_train_predictions, correct_whole_train_predictions = session.run([predictions, correct_predictions],
                                                                             feed_dict=feed_dict_whole_train)

                    # Pintar por pantalla. Precision del TRAIN ENTERO y del MINIBATCH, Precision y F1 de TEST
                    step_train_acc = np.mean(correct_train_predictions)
                    whole_train_acc = np.mean(correct_whole_train_predictions)
                    step_f1_score = metrics.f1_score(np.array(list(itertools.chain.from_iterable(validation_y))), test_predictions + 1, average='macro')
                    print("---------- GD STEP %i of %i ----------" % (step, config.num_steps_GD))
                    print("Minibatch Loss:      {:.5f}".format(l))
                    print("Minibatch Train Acc: {:.5f}".format(step_train_acc))
                    print("Whole Training Acc:  {:.5f}".format(whole_train_acc))
                    print("Validation Accuracy: {:.5f}".format(step_valid_acc))
                    print("Validation F1-Score: {:.5f}".format(step_f1_score))

                last_step_val_acc = step_valid_acc

            ## FIN DE LA OPTIMIZACION EN LA ITERACION DE LA CV.
            # SI EL ULTIMO STEP NO ES EL MEJOR, QUEDARSE CON EL MEJOR MODELO OBTENIDO DURANTE LA OPTIMIZACION
            if best_val_step.step_no != config.num_steps_GD - 1:
                assign_w1 = weights_str['c1'].assign(best_val_step.weight1)
                assign_w2 = weights_str['c2'].assign(best_val_step.weight2)
                assign_w3 = weights_str['d1'].assign(best_val_step.weight3)
                assign_w4 = weights_str['out'].assign(best_val_step.weight4)

                assign_b1 = biases_str['bc1'].assign(best_val_step.biases1)
                assign_b2 = biases_str['bc2'].assign(best_val_step.biases2)
                assign_b3 = biases_str['bd1'].assign(best_val_step.biases3)
                assign_b4 = biases_str['out'].assign(best_val_step.biases4)

                session.run([assign_w1, assign_w2, assign_w3, assign_w4, assign_b1, assign_b2, assign_b3, assign_b4])
                print("*** Resetting model to best optimization step's weights: " + str(best_val_step.step_no) + " ***")

            # CALCULAR MEDIDAS DE CALIDAD PARA TRAINING Y TEST DE LA C.V
            test_predictions, correct_test_predictions = session.run([predictions, correct_predictions],
                                                                     feed_dict=feed_dict_test)

            partial_cv_f1_score = metrics.f1_score(np.array(list(itertools.chain.from_iterable(validation_y))), test_predictions + 1, average='macro')
            partial_cv_valid_acc = np.mean(correct_test_predictions)
            cv_train_acc += best_val_step.train_acc
            cv_valid_acc += partial_cv_valid_acc
            cv_f1_score += partial_cv_f1_score

            print("////////// CROSS.VAL %i of %i FINISHED //////////" % (k_index, k_cross_validation))
            print("Number of training epochs: " + str(np.shape(epochs_training_x)))
            print("Loss: {:.5f}".format(l))
            print("Training accuracy: {:.5f}".format(best_val_step.train_acc))
            print("Validation accuracy: {:.5f}".format(partial_cv_valid_acc))
            print("Validation F1-Score: {:.5f}".format(partial_cv_f1_score))
            k_index += 1


        cv_valid_acc /= k_cross_validation
        cv_f1_score /= k_cross_validation
        cv_train_acc /= k_cross_validation
        print("////////// FINISHED //////////")
        print("Training average accuracy: {:.5f}".format(cv_train_acc))
        print("Validation average accuracy: {:.5f}".format(cv_valid_acc))
        print("Validation average F1-Score: {:.5f}".format(cv_f1_score))

