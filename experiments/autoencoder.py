import tensorflow as tf
import numpy as np
import config
import util
from signal_tools import divide_in_epochs, join_stages_hot_enc
from sklearn import metrics
import itertools

len_training_x = 0

def run(signals, stages):
    # Pre-training Parameters
    learning_rate = 0.001
    pre_training_iterations = 40
    batch_size = 512

    n_hidden_1 = 6000
    n_hidden_2 = 5000
    n_input = config.sample_freq * config.epoch_duration_s * util.get_number_of_channels()
    n_classes = config.num_labels

    # TF Graph
    X = tf.placeholder(tf.float32, [None, n_input])
    Y_labels = tf.placeholder(tf.float32, [None, n_classes])

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'encoder_out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_out': tf.Variable(tf.random_normal([n_classes])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input]))
    }


    def encoder(x):
        # Encodder layer 1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Encodder layer 2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        return layer_2

    def decoder(x):
        # Decoder layer 1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder layer 2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        return layer_2

    def enc_fully_connected(last_encoder_layer):
        # Encoder last layer
        layer_out = tf.matmul(last_encoder_layer, weights['encoder_out']) + biases['encoder_out']
        return layer_out


    ## UNSUPERVISED PRE-TRAINING
    # Build model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Predictions
    Y_pred = decoder_op
    # Original values
    Y = X

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y_pred, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    ## SUPERVISED FINE-TUNING
    # Loss and optimizer
    fully_connected = enc_fully_connected(encoder_op)
    loss_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fully_connected, Y_labels))

    regularizers = (tf.nn.l2_loss(weights['encoder_h1']) + tf.nn.l2_loss(biases['encoder_b1']) +
                    tf.nn.l2_loss(weights['encoder_h2']) + tf.nn.l2_loss(biases['encoder_b2']) +
                    tf.nn.l2_loss(weights['encoder_out']) + tf.nn.l2_loss(biases['encoder_out']))


    # loss_sup += 5e-4 * regularizers

    optimizer_sup = tf.train.AdamOptimizer(learning_rate).minimize(loss_sup)
    # global_step = tf.Variable(0)  # count the number of steps taken.
    # learning_rate = tf.train.exponential_decay(0.05, global_step, 100000, 0.96)
    # optimizer_sup = tf.train.AdamOptimizer(learning_rate).minimize(loss_sup, global_step=global_step)

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    # batch = tf.Variable(0)
    #
    # learning_rate = tf.train.exponential_decay(
    #     0.01,  # Base learning rate.
    #     batch * config.SGD_batch_size,  # Current index into the dataset.
    #     len_training_x,  # Decay step.
    #     0.95,  # Decay rate.
    #     staircase=True)
    # # Use simple momentum for the optimization.
    # optimizer_sup = tf.train.MomentumOptimizer(learning_rate,
    #                                        0.9).minimize(loss_sup,
    #                                                      global_step=batch)

    # Predictions
    predictions = tf.arg_max(fully_connected, 1)
    correct_predictions = tf.equal(tf.argmax(fully_connected, 1), tf.argmax(Y_labels, 1))


    # Initialize variables
    init = tf.initialize_all_variables()

    # Run graph
    with tf.Session() as sess:
        sess.run(init)
        print("Deep Autoencoder with fully-connected layer: Initialized")
        k_cross_validation = util.get_valid_k_cross_validation(len(signals),
                                                           config.k_cross_valid)
        cv_train_acc = 0
        cv_valid_acc = 0
        cv_f1_score = 0
        k_index = 1
        for training_x, training_y, validation_x, validation_y \
            in util.k_fold_cross_validation(signals, stages, k_cross_validation):
            # PRE-TRAINING
            tf.initialize_all_variables().run()
            l_pre = np.inf
            epochs_size_list = [len(x) for x in training_y]
            data_training_x = divide_in_epochs(training_x, epochs_size_list,
                                               config.sample_freq, config.epoch_duration_s)
            data_training_x, training_y = util.down_balance(data_training_x, np.array(list(itertools.chain.from_iterable(training_y))))
            data_training_y = join_stages_hot_enc(training_y, join=False)
            global len_training_x
            len_training_x = len(data_training_x)

            total_batch = int(len(data_training_x) / batch_size)
            print len(signals)
            print total_batch
            print("PRE-TRAINING Optimization: Started")
            for it in range(pre_training_iterations):
                for i in range(total_batch):
                    batch_x, _ = get_batch(data_training_x, data_training_x, batch_size)
                    _, l_pre = sess.run([optimizer, loss], feed_dict={X: batch_x})
                print("Autoencoder Pre-Training iteration: " + str(it) + ". Loss = {:.4f}".format(l_pre))

            print("PRE-TRAINING Optimization: Finished")

            print("FINE-TUNING Classification layer: Started")
            l = np.inf
            labels_size_validation_list = [len(x) for x in validation_y]
            epochs_validation_x = divide_in_epochs(validation_x, labels_size_validation_list,
                                                   config.sample_freq, config.epoch_duration_s)
            epochs_validation_y = join_stages_hot_enc(validation_y)
            feed_dict_test = {X: epochs_validation_x, Y_labels: epochs_validation_y}
            for step in range(config.num_steps_GD):
                ## PREPARAR EPOCHS DE 30 SEGUNDOS. INICIALMENTE POR PACIENTES SEPARADAMENTE
                ## ESTRUCTURA DE ETIQUETAS EN 1-HOT ENCODING
                ## CONJUNTOS POR PACIENTES: FUSIONAR DATOS
                ## TODO: BALANCEO DE CLASES

                offset = (step * config.SGD_batch_size) % (data_training_y.shape[0] - config.SGD_batch_size)
                batch_data = data_training_x[offset:(offset + config.SGD_batch_size), :]
                batch_labels = data_training_y[offset:(offset + config.SGD_batch_size), :]

                feed_dict_train = {X: batch_data, Y_labels: batch_labels}

                _, l, correct_train_predictions = sess.run(
                    [optimizer_sup, loss_sup, correct_predictions], feed_dict=feed_dict_train)

                if step % 10 == 0:
                    # CALCULAR MEDIDAS DE CALIDAD PARA TRAINING Y TEST EN ESTE STEP
                    test_predictions, correct_test_predictions = sess.run([predictions, correct_predictions],
                                                                             feed_dict=feed_dict_test)
                    step_f1_score = metrics.f1_score(validation_y[0], test_predictions + 1, average='macro')
                    step_train_acc = tf.reduce_mean(tf.cast(correct_train_predictions, "float"))
                    step_valid_acc = tf.reduce_mean(tf.cast(correct_test_predictions, "float"))
                    print("----------SGD STEP %i of %i FINISHED----------" % (step, config.num_steps_GD))
                    print("Loss: {:.5f}".format(l))
                    print("Training accuracy: {:.5f}".format(step_train_acc.eval()))
                    print("Validation accuracy: {:.5f}".format(step_valid_acc.eval()))
                    print("Validation F1-Score: {:.5f}".format(step_f1_score))

            print("FINE-TUNING Classification layer: Finished")
            # CALCULAR MEDIDAS DE CALIDAD PARA TRAINING Y TEST DEL CV
            test_predictions, correct_test_predictions = sess.run([predictions, correct_predictions],
                                                                     feed_dict=feed_dict_test)
            partial_cv_f1_score = metrics.f1_score(validation_y[0], test_predictions + 1, average='macro')
            partial_cv_valid_acc = tf.reduce_mean(tf.cast(correct_test_predictions, "float"))
            cv_valid_acc += partial_cv_valid_acc
            cv_f1_score += partial_cv_f1_score


            print("////////// CROSS.VAL %i of %i FINISHED //////////" % (k_index, k_cross_validation))
            print("Loss: {:.5f}".format(l))
            print("Validation accuracy: {:.5f}".format(partial_cv_valid_acc.eval()))
            print("Validation F1-Score: {:.5f}".format(partial_cv_f1_score))
            k_index += 1

        cv_valid_acc /= k_cross_validation
        cv_f1_score /= k_cross_validation
        print("////////// FINISHED //////////")
        print("Validation average accuracy: {:.5f}".format(cv_valid_acc.eval()))
        print("Validation average F1-Score: {:.5f}".format(cv_f1_score))


def get_batch(X, X2, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X2[a]