import tensorflow as tf
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
        n_hidden_1 = 25
        n_hidden_2 = 25
        n_input = config.sample_freq * config.epoch_duration_s * util.get_number_of_channels()
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

            # DROPOUT (TESTING PURPOSES)
            # layer_2 = tf.nn.dropout(layer_2, 0.10, seed=None)

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

        regularizers = (tf.nn.l2_loss(weights_str['h1']) + tf.nn.l2_loss(biases_str['b1']) +
                        tf.nn.l2_loss(weights_str['h2']) + tf.nn.l2_loss(biases_str['b2']) +
                        tf.nn.l2_loss(weights_str['out']) + tf.nn.l2_loss(biases_str['out']))

        # loss += 5e-5 * regularizers

        global_step = tf.Variable(0)
        #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        optimizer = tf.train.AdamOptimizer(config.base_learning_rate).minimize(loss, global_step=global_step)

        # PREDICTIONS
        predictions = tf.arg_max(pred, 1)
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(tf_data_y, 1))


    ### LANZAMIENTO DE LA SESION
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Gradient Descent MultiLayer-Perceptron: Initialized")
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

            epochs_training_x, training_y = util.down_balance(epochs_training_x,
                                                            np.array(list(itertools.chain.from_iterable(training_y))))

            epochs_training_y = join_stages_hot_enc(training_y, join=False)
            ## SHUFFLE EPOCHS AFTER BALANCING ##
            epochs_training_x, epochs_training_y = shuffle(epochs_training_x, epochs_training_y)

            feed_dict_train = {tf_data_x: epochs_training_x, tf_data_y: epochs_training_y}
            labels_size_validation_list = [len(x) for x in validation_y]
            epochs_validation_x = divide_in_epochs(validation_x, labels_size_validation_list,
                                                   config.sample_freq, config.epoch_duration_s)
            epochs_validation_y = join_stages_hot_enc(validation_y)
            feed_dict_test = {tf_data_x: epochs_validation_x, tf_data_y: epochs_validation_y}
            for step in range(config.num_steps_GD):
                # GD: OPTIMIZAR MODELO PARA DATOS DE TRAIN
                _, l, correct_train_predictions = session.run(
                    [optimizer, loss, correct_predictions], feed_dict=feed_dict_train)

                # CALCULAR MEDIDAS DE CALIDAD PARA TRAINING Y TEST DE LA C.V
                test_predictions, correct_test_predictions = session.run([predictions, correct_predictions],
                                                                         feed_dict=feed_dict_test)

                step_valid_acc = tf.reduce_mean(tf.cast(correct_test_predictions, "float")).eval()
                if step_valid_acc > best_val_step.val_acc:
                    best_val_step.set(step, tf.reduce_mean(tf.cast(correct_train_predictions, "float")).eval(),
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

                if step % 10 == 0:
                    # Pintar por pantalla
                    step_f1_score = metrics.f1_score(np.array(list(itertools.chain.from_iterable(validation_y))), test_predictions + 1, average='macro')
                    step_train_acc = tf.reduce_mean(tf.cast(correct_train_predictions, "float"))
                    print("---------- GD STEP %i of %i ----------" % (step, config.num_steps_GD))
                    print("Loss: {:.5f}".format(l))
                    print("Training accuracy: {:.5f}".format(step_train_acc.eval()))
                    print("Validation accuracy: {:.5f}".format(step_valid_acc))
                    print("Validation F1-Score: {:.5f}".format(step_f1_score))

                last_step_val_acc = step_valid_acc

            ## FIN DE LA OPTIMIZACION EN LA ITERACION DE LA CV.
            # SI EL ULTIMO STEP NO ES EL MEJOR, QUEDARSE CON EL MEJOR MODELO OBTENIDO DURANTE LA OPTIMIZACION
            if best_val_step.step_no != config.num_steps_GD - 1:
                assign_w1 = weights_str['h1'].assign(best_val_step.weight1)
                assign_w2 = weights_str['h2'].assign(best_val_step.weight2)
                assign_w3 = weights_str['out'].assign(best_val_step.weight3)

                assign_b1 = biases_str['b1'].assign(best_val_step.biases1)
                assign_b2 = biases_str['b2'].assign(best_val_step.biases2)
                assign_b3 = biases_str['out'].assign(best_val_step.biases3)

                session.run([assign_w1, assign_w2, assign_w3, assign_b1, assign_b2, assign_b3])
                print("*** Resetting model to best optimization step's weights: " + str(best_val_step.step_no) + " ***")

            # CALCULAR MEDIDAS DE CALIDAD PARA TRAINING Y TEST DE LA C.V
            test_predictions, correct_test_predictions = session.run([predictions, correct_predictions],
                                                                     feed_dict=feed_dict_test)

            partial_cv_f1_score = metrics.f1_score(np.array(list(itertools.chain.from_iterable(validation_y))), test_predictions + 1, average='macro')
            partial_cv_valid_acc = tf.reduce_mean(tf.cast(correct_test_predictions, "float"))
            cv_train_acc += best_val_step.train_acc
            cv_valid_acc += partial_cv_valid_acc
            cv_f1_score += partial_cv_f1_score

            print("////////// CROSS.VAL %i of %i FINISHED //////////" % (k_index, k_cross_validation))
            print("Number of training epochs: " + str(np.shape(epochs_training_x)))
            print("Loss: {:.5f}".format(l))
            print("Training accuracy: {:.5f}".format(best_val_step.train_acc))
            print("Validation accuracy: {:.5f}".format(partial_cv_valid_acc.eval()))
            print("Validation F1-Score: {:.5f}".format(partial_cv_f1_score))
            k_index += 1


        cv_valid_acc /= k_cross_validation
        cv_f1_score /= k_cross_validation
        cv_train_acc /= k_cross_validation
        print("////////// FINISHED //////////")
        print("Training average accuracy: {:.5f}".format(cv_train_acc))
        print("Validation average accuracy: {:.5f}".format(cv_valid_acc))
        print("Validation average F1-Score: {:.5f}".format(cv_f1_score))

