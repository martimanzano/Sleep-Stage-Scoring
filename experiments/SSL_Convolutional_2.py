import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import tflearn as tflearn
from best_step import BestStep
from signal_tools import *
from sklearn.utils import shuffle
from patient_stats import *

def run(signals, stages):
### CREACION DE GRAFO DE COMPUTO
    graph = tf.Graph()
    with graph.as_default():
        ### NETWORK PARAMETERS
        ## CONVOLUTIONAL PART
        # CONVOLUTION PARAMETERS
        conv_in_width_1 = 8
        conv_in_width_2 = 16
        # conv_in_width_3 = 16

        conv_out_depth_1 = 9
        conv_out_depth_2 = 18
        # conv_out_depth_3 = 64

        stride_conv_1 = 2
        stride_conv_2 = 2
        stride_conv_3 = 1

        # POOLINGS
        pool_in_width_1 = 8
        pool_in_width_2 = 8
        # pool_in_width_3 = 2

        stride_pool_1 = 2
        stride_pool_2 = 2
        stride_pool_3 = 1

        # FULLY CONNECTED
        n_hidden_fc_1 = 2500#2500
        # n_hidden_fc_2 = 1000

        ## AUTOENCODER PART
        n_hidden_auto_1 = 1600#1600
        # n_hidden_auto_2 = 2000

        ## CONCATENATE LAYER
        n_concatenate = 200


        n_input = current_sample_freq_t() * config.epoch_duration_s * util.get_number_of_channels()
        n_input_freq = (current_sample_freq_f() * config.epoch_duration_s * util.get_number_of_channels()) / 2
        n_classes = config.num_labels

        ## INPUT DATA. RECEIVED FROM THE SESSION
        time_data_x = tf.placeholder(tf.float32, [None, n_input])   #shape=(config.SGD_batch_size, n_input))
        freq_data_x = tf.placeholder(tf.float32, [None, n_input_freq])
        is_training_ph = tf.placeholder(tf.bool)

        tf_data_y = tf.placeholder(tf.float32, [None, n_classes]) #shape=(config.SGD_batch_size, n_classes))

        keep_prob = tf.placeholder(tf.float32)


        def conv1d(x, W, b, strides_conv):
            x = tf.nn.conv1d(x, W, stride=strides_conv, padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

        def maxpool1d(x, pool_in_width, stride_pool):
            return tflearn.layers.conv.max_pool_1d(x, pool_in_width, strides=stride_pool, padding='SAME')
            # return tflearn.layers.conv.avg_pool_1d(incoming=x, kernel_size=pool_in_width, strides=stride_pool, padding='SAME')

        ## MODEL SET UP
        def convolutional_net(x_data, weights, biases, dropout_prob, multiscale):
            # Convolution layer 1
            with tf.variable_scope('conv0') as vs:
                print(x_data.get_shape())
                x_data = tf.reshape(x_data, [-1, n_input, 1])
                print(x_data.get_shape())
                x_data = tf.cond(is_training_ph,
                                       lambda: tf.contrib.layers.batch_norm(x_data, is_training=True,
                                                                            center=True, scale=True, decay=0.9,
                                                                            activation_fn=tf.nn.relu,
                                                                            updates_collections=None, scope=vs),
                                       lambda: tf.contrib.layers.batch_norm(x_data, is_training=False,
                                                                            center=True, scale=True, decay=0.9,
                                                                            activation_fn=tf.nn.relu,
                                                                            updates_collections=None, scope=vs,
                                                                            reuse=True))
            with tf.variable_scope('conv1') as vs:
                conv = conv1d(x_data, weights['c1'], biases['bc1'], strides_conv=stride_conv_1)
                print(conv.get_shape())
                # Max Pooling 1
                conv = maxpool1d(conv, pool_in_width_1, stride_pool_1)
                print(conv.get_shape())

                conv = tf.cond(is_training_ph,
                                       lambda: tf.contrib.layers.batch_norm(conv, is_training=True,
                                                                            center=True, scale=True, decay=0.9,
                                                                            activation_fn=tf.nn.relu,
                                                                            updates_collections=None, scope=vs),
                                       lambda: tf.contrib.layers.batch_norm(conv, is_training=False,
                                                                            center=True, scale=True, decay=0.9,
                                                                            activation_fn=tf.nn.relu,
                                                                            updates_collections=None, scope=vs,
                                                                            reuse=True))

            # Convolution layer 2
            with tf.variable_scope('conv2') as vs2:
                conv = conv1d(conv, weights['c2'], biases['bc2'], strides_conv=stride_conv_2)
                print(conv.get_shape())
                # Max Pooling 2
                conv = maxpool1d(conv, pool_in_width_2, stride_pool_2)
                print(conv.get_shape())

                conv = tf.cond(is_training_ph,
                               lambda: batch_norm(conv, is_training=True, center=True, scale=True, decay=0.9,
                                                                    activation_fn=tf.nn.relu,
                                                                    updates_collections=None, scope=vs2),
                               lambda: batch_norm(conv, is_training=False, center=True, scale=True, decay=0.9,
                                                                    activation_fn=tf.nn.relu,
                                                                    updates_collections=None, scope=vs2,
                                                                    reuse=True))


            # Convolution layer 3
            # conv = conv1d(conv, weights['c3'], biases['bc3'], strides_conv=stride_conv_3)
            # print(conv.get_shape())
            # # # # Max Pooling 3
            # conv = maxpool1d(conv, pool_in_width_3, stride_pool_3)
            # print(conv.get_shape())

            # Fully connected layer 1
            with tf.variable_scope('fc1') as vs3:
                fc = tf.reshape(conv, [-1, weights_conv['d1'].get_shape().as_list()[0]])
                print(fc.get_shape())

                ############ END MULTISCALE PART ############
                if multiscale:
                    fc = tf.nn.dropout(fc, dropout_prob)
                    return fc

                fc = tf.add(tf.matmul(fc, weights['d1']), biases['bd1'])
                # fc = tf.add(tf.matmul(x_data, weights['d1']), biases['bd1'])
                fc = tf.nn.relu(fc)
                print("shape fc cond: " + str(fc.get_shape()))

                fc = tf.cond(is_training_ph,
                             lambda: batch_norm(fc, is_training=True, center=True, scale=True, decay=0.9,
                                                activation_fn=None,
                                                updates_collections=None, scope=vs3),
                             lambda: batch_norm(fc, is_training=False, center=True, scale=True, decay=0.9,
                                                activation_fn=None,
                                                updates_collections=None, scope=vs3,
                                                reuse=True))

                # Fully connected layer 2
                # fc2 = tf.add(tf.matmul(fc, weights['d2']), biases['bd2'])
                # fc2 = tf.nn.relu(fc2)
                # print(fc2.get_shape())
                # Apply Dropout regularization
                fc = tf.nn.dropout(fc, dropout_prob)

                # Output -- Class prediction
                out = tf.add(tf.matmul(fc, weights['out']), biases['out'])

                print("shape original fc: " + str(fc.get_shape()))
                print("shape out: " + str(out.get_shape()))
                print multiscale
                # return tf.cond(multiscale, lambda: tf.nn.dropout(fc, dropout_prob), lambda: out)
                return out



        def encoder(x, dropout_prob, multiscale):
            # Encodder layer 1
            with tf.variable_scope('enc1') as vs4:
                layer_1 = tf.cond(is_training_ph,
                                 lambda: batch_norm(x, is_training=True, center=True, scale=True, decay=0.9,
                                                    activation_fn=None,
                                                    updates_collections=None, scope=vs4),
                                 lambda: batch_norm(x, is_training=False, center=True, scale=True, decay=0.9,
                                                    activation_fn=None,
                                                    updates_collections=None, scope=vs4,
                                                    reuse=True))


                layer_1 = tf.add(tf.matmul(layer_1, weights_enc['encoder_h1']),
                                               biases_enc['encoder_b1'])
                # print multiscale
                # return tf.cond(multiscale, lambda: layer_1, lambda: tf.nn.relu(layer_1))
                if multiscale:
                    return layer_1
                else:
                    return tf.nn.relu(layer_1)


        def decoder(x):
            # Decoder layer 1
            # layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights_enc['decoder_h1']),
            #                                 biases_enc['decoder_b1']))
            # Decoder layer 2
            layer_2 = tf.nn.relu(tf.add(tf.matmul(x, weights_enc['decoder_h2']),
                                               biases_enc['decoder_b2']))
            return layer_2


        def enc_fully_connected(last_encoder_layer, dropout_prob):
            # Encoder last layer
            # Apply Dropout regularization
            last_encoder_layer = tf.nn.dropout(last_encoder_layer, dropout_prob)
            layer_out = tf.matmul(last_encoder_layer, weights_enc['encoder_out']) + biases_enc['encoder_out']
            return layer_out


        # WEIGHTS AND BIASES SET UP
        weights_conv = {         # 4D: [height, width, in_ch, out_ch]
            'c1': tf.Variable(tf.random_normal([conv_in_width_1, 1, conv_out_depth_1])),
            'c2': tf.Variable(tf.random_normal([conv_in_width_2, conv_out_depth_1, conv_out_depth_2])),
            # 'c3': tf.Variable(tf.random_normal([conv_in_width_3, conv_out_depth_2, conv_out_depth_3])),
            'd1': tf.Variable(tf.random_normal([conv_out_depth_2 *
                                                 (n_input / (stride_pool_1 * stride_pool_2 * stride_pool_3 *
                                                            stride_conv_1 * stride_conv_2 * stride_conv_3)),
                                                                                                     n_hidden_fc_1])),
            # 'd1': tf.Variable(tf.random_normal([n_input, n_hidden_fc_1])),
            # 'd2': tf.Variable(tf.random_normal([n_hidden_fc_1, n_hidden_fc_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_fc_1, n_classes]))
        }
        biases_conv = {
            'bc1': tf.Variable(tf.random_normal([conv_out_depth_1])),
            'bc2': tf.Variable(tf.random_normal([conv_out_depth_2])),
            # 'bc3': tf.Variable(tf.random_normal([conv_out_depth_3])),
            'bd1': tf.Variable(tf.random_normal([n_hidden_fc_1])),
            # 'bd2': tf.Variable(tf.random_normal([n_hidden_fc_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        weights_enc = {
            # 'encoder_h1': tf.Variable(tf.random_normal([n_input_freq, n_hidden_auto_1])),
            'encoder_h1': tf.get_variable("encoder_h1", shape = [n_input_freq, n_hidden_auto_1],
                                          initializer = tf.contrib.layers.xavier_initializer()),
            # 'encoder_h2': tf.Variable(tf.random_normal([n_hidden_auto_1, n_hidden_auto_2])),
            # 'encoder_out': tf.Variable(tf.random_normal([n_hidden_auto_1, n_classes])),
            'encoder_out': tf.get_variable("encoder_out", shape=[n_hidden_auto_1, n_classes],
                                          initializer=tf.contrib.layers.xavier_initializer()),
            # 'decoder_h1': tf.Variable(tf.random_normal([n_hidden_auto_2, n_hidden_auto_1])),
            # 'decoder_h2': tf.Variable(tf.random_normal([n_hidden_auto_1, n_input_freq]))
            'decoder_h2': tf.get_variable("decoder_h2", shape=[n_hidden_auto_1, n_input_freq],
                                           initializer=tf.contrib.layers.xavier_initializer()),
        }

        biases_enc = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_auto_1])),
            # 'encoder_b2': tf.Variable(tf.random_normal([n_hidden_auto_2])),
            'encoder_out': tf.Variable(tf.random_normal([n_classes])),
            # 'decoder_b1': tf.Variable(tf.random_normal([n_hidden_auto_1])),
            'decoder_b2': tf.Variable(tf.random_normal([n_input_freq]))
        }

        multiscale_str = {
            # 'w_out': tf.get_variable("w_out", shape=[n_hidden_auto_1 + n_hidden_fc_1, n_classes],
            #                          initializer=tf.contrib.layers.xavier_initializer()),
            'w_concat': tf.get_variable("w_concat", shape=[(conv_out_depth_2 * (n_input / (stride_pool_1 * stride_pool_2 *
                                              stride_pool_3 * stride_conv_1 * stride_conv_2 * stride_conv_3))) + n_hidden_auto_1,
                                               n_concatenate], initializer=tf.contrib.layers.xavier_initializer()),
            # 'w_out': tf.get_variable("w_out", shape=[n_hidden_auto_1 + n_hidden_fc_1, n_classes],
            #                                                    initializer=tf.contrib.layers.xavier_initializer()),
            'w_out': tf.get_variable("w_out", shape=[n_concatenate, n_classes],
                                     initializer=tf.contrib.layers.xavier_initializer()),
            'b_concat': tf.Variable(tf.random_normal([n_concatenate])),
            'b_out': tf.Variable(tf.random_normal([n_classes]))
        }

        ### DNN
        ## UNSUPERVISED PRE-TRAINING
        # BUILD AUTOENCODER MODEL

        with tf.variable_scope("freq_ops") as scope:
            encoder_op = encoder(freq_data_x, keep_prob, multiscale= False)
            scope.reuse_variables()
            ##decoder_op = decoder(encoder_op)

            # Predictions
            ##Y_pred = decoder_op
            # Original values
            ## Y = freq_data_x

            ## loss_pre = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y_pred, Y))
            ##optimizer_pre = tf.train.AdamOptimizer(0.01).minimize(loss_pre)

            ## SUPERVISED TRAINING
            pred_f = enc_fully_connected(encoder_op, keep_prob)
            loss_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_f, tf_data_y))

            optimizer_f = tf.train.AdamOptimizer(config.base_learning_rate).minimize(loss_f)

            # LOWER PART: FREQ DOMAIN WITH DNN (optional autoencoder pretraining)
            freq_layer = encoder(freq_data_x, keep_prob, multiscale=True)
            freq_layer = tf.nn.dropout(freq_layer, keep_prob)


        ### CNN
        with tf.variable_scope("time_ops") as scope:
            pred_t = convolutional_net(time_data_x, weights_conv, biases_conv, keep_prob, multiscale= False)
            scope.reuse_variables()
            # LOSS AND OPTIMIZER SET UP
            loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_t, tf_data_y))
            #
            # # regularizers = (tf.nn.l2_loss(weights_str['c1']) + tf.nn.l2_loss(biases_str['bc1']) +
            # #                 tf.nn.l2_loss(weights_str['d1']) + tf.nn.l2_loss(biases_str['bd1']) +
            # #                 tf.nn.l2_loss(weights_str['out']) + tf.nn.l2_loss(biases_str['out']))
            # #
            # # regularizers = (tf.nn.l2_loss(weights_str['d1']) + tf.nn.l2_loss(biases_str['bd1']) +
            # #                 tf.nn.l2_loss(weights_str['out']) + tf.nn.l2_loss(biases_str['out']))
            # #
            # # loss += 5e-5 * regularizers
            #
            optimizer_t = tf.train.AdamOptimizer(config.base_learning_rate).minimize(loss_t)


            # UPPER PART: TIME DOMAIN WITH CNN
            #time_layer = pred_t
            time_layer = convolutional_net(time_data_x, weights_conv, biases_conv, keep_prob, multiscale= True)

        ## MULTI-SCALE: CONCATENATE BOTH NETWORK'S PARTS
        # CONCATENATE PART
        # FC1
        with tf.variable_scope('cp1') as vs5:
            concat_layer = tf.concat(concat_dim=1, values=[time_layer, freq_layer])
            print("shape TIME:" + str(time_layer.get_shape()))
            print("shape FREQ:" + str(freq_layer.get_shape()))
            print("shape MERGE:" + str(concat_layer.get_shape()))
            concat_layer = tf.nn.relu(concat_layer)
            concat_layer = tf.cond(is_training_ph,
                              lambda: batch_norm(concat_layer, is_training=True, center=True, scale=True, decay=0.9,
                                                 activation_fn=None, updates_collections=None, scope=vs5),
                              lambda: batch_norm(concat_layer, is_training=False, center=True, scale=True, decay=0.9,
                                                 activation_fn=None, updates_collections=None, scope=vs5,
                                                 reuse=True))

        # FC2
        with tf.variable_scope('cp2') as vs6:
            concat_layer = tf.add(tf.matmul(concat_layer, multiscale_str['w_concat']), multiscale_str['b_concat'])
            concat_layer = tf.nn.relu(concat_layer)
            concat_layer = tf.cond(is_training_ph,
                               lambda: batch_norm(concat_layer, is_training=True, center=True, scale=True, decay=0.9,
                                                  activation_fn=None, updates_collections=None, scope=vs6),
                               lambda: batch_norm(concat_layer, is_training=False, center=True, scale=True, decay=0.9,
                                                  activation_fn=None, updates_collections=None, scope=vs6,
                                                  reuse=True))

            print(concat_layer.get_shape())
            concat_layer = tf.nn.dropout(concat_layer, keep_prob)

        # LOGITS MULTISCALE
        pred_ms = tf.add(tf.matmul(concat_layer, multiscale_str['w_out']), multiscale_str['b_out'])
        # LOSS MULTISCALE
        loss_ms = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_ms, tf_data_y))
        # OPTIMIZER MULTISCALE
        global_step = tf.Variable(0, trainable=False)
        optimizer_ms = tf.train.AdamOptimizer(config.base_learning_rate).minimize(loss_ms, global_step=global_step)

        # LOSS, OPTIMIZER AND PREDICTIONS SELECTION: ONLY A NETWORK'S PART OR THE WHOLE MULTISCALE NETWORK
        loss = loss_ms
        optimizer = optimizer_ms
        pred = pred_ms

        # PREDICTIONS
        predictions = tf.arg_max(pred, 1)
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(tf_data_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    ######################### GRAPH DEFINITION END #############################

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
        patient_stats = Patient_stats()
        test_predictions = None

        ## CROSS VALIDATION MAIN LOOP. WORKS WITH A SET OF PATIENT'S SIGNALS FOR TRAINING AND VALIDATION
        for training_x, training_y, validation_x, validation_y \
                in util.k_fold_cross_validation(signals, stages, k_cross_validation, randomise=config.randomize_patients):
            # SET UP CONTROL VARIABLES
            tf.initialize_all_variables().run() # RESET OPTIMIZATION FOR EVERY C.V ITERATION
            l = np.inf
            best_step = BestStep(0, 0, 0, 0, 0, 0)
            early_stop_remaining_steps = config.GD_early_stopping_steps

            print("////////// CROSS.VAL %i of %i STARTED //////////" % (k_index, k_cross_validation))
            # SET UP DATA VARIABLES
            ## PREPARAR EPOCHS DE 30 SEGUNDOS. INICIALMENTE POR PACIENTES SEPARADAMENTE
            ## ESTRUCTURA DE ETIQUETAS EN 1-HOT ENCODING
            ## CONJUNTOS POR PACIENTES: FUSIONAR DATOS

            # TRAINING
            labels_size_training_list = [len(x) for x in training_y]
            epochs_training_x = divide_in_epochs(training_x, labels_size_training_list,
                                                 config.sample_freq, config.epoch_duration_s)
            if config.balance_classes:
                epochs_training_x, training_y = util.down_balance(epochs_training_x,
                                                            np.array(list(itertools.chain.from_iterable(training_y))))
                force_join_stages = False
            else:
                force_join_stages = True
            epochs_training_y = join_stages_hot_enc(training_y, join=force_join_stages)
            ## SHUFFLE EPOCHS AFTER BALANCING ##
            epochs_training_x, epochs_training_y = shuffle(epochs_training_x, epochs_training_y)

            # VALIDATION
            labels_size_validation_list = [len(x) for x in validation_y]
            epochs_validation_x = divide_in_epochs(validation_x, labels_size_validation_list,
                                                   config.sample_freq, config.epoch_duration_s)
            epochs_validation_y = join_stages_hot_enc(validation_y)

            # FREQUENCY DOMAIN STUFF
            # freq-factor downsampling
            freq_epochs_training_x = downsample_epochs_set(epochs_training_x, config.downsampling_factor_freq)
            freq_epochs_training_x = get_freq_domain_epochs(freq_epochs_training_x)

            freq_epochs_validation_x = downsample_epochs_set(epochs_validation_x, config.downsampling_factor_freq)
            freq_epochs_validation_x = get_freq_domain_epochs(freq_epochs_validation_x)
            # freq_epochs_training_x = preprocessing.scale(freq_epochs_training_x)
            # freq_epochs_validation_x = preprocessing.scale(freq_epochs_validation_x

            # TIME DOMAIN STUFF
            epochs_training_x = downsample_epochs_set(epochs_training_x, config.downsampling_factor)
            epochs_validation_x = downsample_epochs_set(epochs_validation_x, config.downsampling_factor)
            # EPOCH SIGNAL NORMALIZATION
            epochs_training_x = preprocess_epochs_set(epochs_training_x)
            epochs_validation_x = preprocess_epochs_set(epochs_validation_x)
            # SSL DATA
            epochs_ssl = util.get_ssl_data_cv_iteration(k_index)
            ssl_data = util.build_ssl_stages_data(epochs_ssl)

            print("Number of training epochs: " + str(np.shape(epochs_training_x)))
            print "         Freq. epoch size: " + str(freq_epochs_training_x.shape[1])
            print "         Time. epoch size: " + str(epochs_training_x.shape[1])

            if config.pretrain:
                ## MODEL PRETRAINING
                print("PRE-TRAINING Optimization: Started")
                # pretrain_data = concatenate((freq_epochs_training_x, freq_epochs_validation_x))
                for step in range(config.pretraining_steps):
                    # PREPARAR MINI BATCH A PARTIR DE LOS EPOCHS DE ENTRENAMIENTO
                    offset = (step * config.SGD_batch_size) % (epochs_training_x.shape[0] - config.SGD_batch_size)
                    batch_data_x = epochs_training_x[offset:(offset + config.SGD_batch_size)]
                    batch_data_f_x = freq_epochs_training_x[offset:(offset + config.SGD_batch_size)]
                    batch_data_y = epochs_training_y[offset:(offset + config.SGD_batch_size)]

                    _, l_t = session.run([optimizer_t, loss_t], feed_dict={time_data_x: batch_data_x,
                                                                           tf_data_y: batch_data_y,
                                                                           keep_prob: config.dropout_keep_rate,
                                                                           is_training_ph:True})
                    _, l_f = session.run([optimizer_f , loss_f], feed_dict={freq_data_x: batch_data_f_x,
                                                                            tf_data_y: batch_data_y,
                                                                            keep_prob: config.dropout_keep_rate,
                                                                            is_training_ph:True})
                    if step % config.print_status_every == 0:
                        print(" Time Pre-Training iteration: " + str(step) + ". Loss = {:.4f}".format(l_t))
                        print("Freq. Pre-Training iteration: " + str(step) + ". Loss = {:.4f}".format(l_f))
                print("PRE-TRAINING Optimization: Finished")

            feed_dict_test = {freq_data_x: freq_epochs_validation_x, time_data_x: epochs_validation_x,
                              tf_data_y: epochs_validation_y, keep_prob: 1, is_training_ph:False}

            ## MODEL OPTIMIZATION
            for step in range(config.num_steps_GD):
                # PREPARAR MINI BATCHS A PARTIR DE LOS EPOCHS DE ENTRENAMIENTO
                offset = (step * config.SGD_batch_size) % (epochs_training_y.shape[0] - config.SGD_batch_size)
                batch_data_x_f = freq_epochs_training_x[offset:(offset + config.SGD_batch_size)]
                batch_data_x_t = epochs_training_x[offset:(offset + config.SGD_batch_size)]

                batch_data_y = epochs_training_y[offset:(offset + config.SGD_batch_size)]
                # DICCIONARIO PARA RELLENAR LOS PLACEHOLDERS
                feed_dict_train = {freq_data_x: batch_data_x_f, time_data_x: batch_data_x_t,
                                   tf_data_y: batch_data_y, keep_prob: config.dropout_keep_rate, is_training_ph:True}

                # MINIBATCH GRADIENT DESCENT: OPTIMIZAR MODELO PARA DATOS DEL MINIBATCH DE TRAIN
                _, l, correct_minib_predictions, minib_accur = session.run(
                    [optimizer, loss, correct_predictions, accuracy], feed_dict=feed_dict_train)

                # CALCULAR MEDIDAS DE CALIDAD PARA LA PARTE DE VALIDACION DE LA C.V
                test_predictions, correct_test_predictions, test_accur = session.run([predictions,
                                                                                      correct_predictions, accuracy],
                                                                         feed_dict=feed_dict_test)
                # COMPROBAR SI ESTE PASO DE OPTIMIZACION MEJORA EL MODELO (EN BASE A LA ACC. DE TEST)
                if test_accur > best_step.val_acc: # SAVE BEST MODEL YET QUALITY METRICS
                    best_step.set(step, minib_accur, l, test_accur, util.get_f1_score(validation_y, test_predictions),
                                  util.get_confusion_matrix(validation_y, test_predictions))
                    early_stop_remaining_steps = config.GD_early_stopping_steps
                else:
                    early_stop_remaining_steps -= 1
                    if early_stop_remaining_steps == 0:
                        print("EARLY-STOPPING OPTIMIZATION DUE TO " + str(config.GD_early_stopping_steps)
                              + " STEPS WITH VALIDATION ACCURACY NOT IMPROVING...")
                        break

                # PINTADO DE DATOS DE LA OPTIMIZACION SI TOCA POR ITERACION O SI HA HABIDO MEJORA
                if step % config.print_status_every == 0 or best_step.step_no == step:
                    # COMO OPTIMIZAMOS POR MINI BATCH, PRIMERO RECOGER PRECISION DE TRAIN PARA EL CONJUNTO ENTERO
                    feed_dict_whole_train = {freq_data_x: freq_epochs_training_x, time_data_x: epochs_training_x,
                                             tf_data_y: epochs_training_y, keep_prob: 1, is_training_ph:False}
                    whole_train_predict, correct_whole_train_predict, whole_train_acc = \
                        session.run([predictions, correct_predictions, accuracy], feed_dict=feed_dict_whole_train)

                    # Pintar por pantalla. Precision del TRAIN ENTERO y del MINIBATCH, Precision y F1 de TEST
                    step_f1_score = util.get_f1_score(validation_y, test_predictions)
                    print("---------- GD STEP %i of %i ----------" % (step, config.num_steps_GD))
                    print("     Minibatch Loss: {:.5f}".format(l))
                    print("Minibatch Train Acc: {:.5f}".format(minib_accur))
                    util.print_main_metrics(whole_train_acc, test_accur, step_f1_score)

                    if best_step.step_no == step:
                        best_step.set_whole_train_acc(whole_train_acc)
                        best_step.set_whole_test_predictions(test_predictions)

                ## FIN DE LA OPTIMIZACION EN LA ITERACION DE LA CV.
                # SI EL ULTIMO STEP NO ES EL MEJOR, QUEDARSE CON EL MEJOR MODELO OBTENIDO DURANTE LA OPTIMIZACION
                #if best_step.step_no != config.num_steps_GD - 1:
                 #   print("*** Resetting model to best optimization step's weights: " + str(best_step.step_no) + " ***")

            ## PAINT BEST STEP'S CROSS VALIDATION QUALITY METRICS
            # BEST STEP QUALITY METRICS SAVED IN THE CHURRO STRUCTURE
            cv_train_acc += best_step.whole_train_acc
            cv_valid_acc += best_step.val_acc
            cv_f1_score += best_step.val_f1

            util.store_patient_ssl_data(k_index, validation_y, test_predictions + 1)

            if best_step.val_acc < patient_stats.worst_val_acc:
                patient_stats.set_worst(k_index, best_step.val_acc)
            if best_step.val_acc > patient_stats.best_val_acc:
                patient_stats.set_best(k_index, best_step.val_acc)

            print("////////// CROSS.VAL %i of %i FINISHED //////////" % (k_index, k_cross_validation))
            print(" *** Best optimization step: " + str(best_step.step_no) + " ***")
            print("Best minibatch loss:      {:.5f}".format(best_step.minibatch_loss))
            util.print_main_metrics(best_step.whole_train_acc, best_step.val_acc, best_step.val_f1)
            print(best_step.confusion_matrix)
            patient_stats.print_stats()
            k_index += 1

        ### FIN DE LA C.V. CALCULAR METRICAS MEDIAS Y PINTARLAS
        cv_valid_acc /= k_cross_validation
        cv_f1_score /= k_cross_validation
        cv_train_acc /= k_cross_validation
        print("////////// FINISHED //////////")
        print "*** AVERAGE METRICS: ***"
        util.print_main_metrics(cv_train_acc, cv_valid_acc, cv_f1_score)
        patient_stats.print_stats()