import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.optimizers import Adam
import tflearn as tfl
import math
import h5py
import numpy as np
from h5_keys import *


def _rmse(pred, targets):
    """
    Cost function
    :param pred: predictions
    :param targets: targets
    :return:
    """
    with tf.name_scope("rmse_loss"):
        return tf.sqrt(tf.reduce_mean(tf.squared_difference(pred, targets)), name="rmse")


def _rmse_valid(pred, targets, inputs):
    """
    Validation function (negative rmse)
    :param pred: predictions
    :param targets: targets
    :param inputs: inputs
    :return:
    """
    with tf.name_scope("rmse_validation"):
        return -_rmse(pred, targets)


def _rmse_test(targets, predictions):
    """
    Computes the absolute error on the prediction on each example
    :param targets: targets
    :param predictions: predictions
    :return: absolute error vector
    """
    return np.sqrt(np.square(np.diff([targets, predictions], axis=0)))[0]


def _nn_creation(first_layer_width, last_layer_width, depth, epsilon=1e-8, learning_rate=0.001, dropout_val=0.99,
                stddev_init=0.001, hidden_act='relu', outlayer_act='linear', weight_decay=0.001,
                validation_fun=_rmse_valid, cost_fun=_rmse, gpu_mem_prop=1):
    """
    Creates a neural network with the given parameters.
    :param first_layer_width:
    :param last_layer_width:
    :param depth:
    :param epsilon:
    :param learning_rate:
    :param dropout_val:
    :param stddev_init:
    :param hidden_act:
    :param outlayer_act:
    :param weight_decay:
    :param validation_fun:
    :param cost_fun:
    :param gpu_mem_prop:
    :return: created neural network
    """

    # Weights initialization
    winit = tfl.initializations.truncated_normal(stddev=stddev_init, dtype=tf.float32, seed=None)

    # GPU memory utilisation proportion
    tfl.init_graph(num_cores=16, gpu_memory_fraction=gpu_mem_prop, soft_placement=True)

    # Creating NN input
    network = input_data(shape=[None, first_layer_width], name='input')

    # Calculating width coef
    width_coef = (last_layer_width - first_layer_width) / (depth - 1)

    # Creating hidden layers
    for i in range(depth):

        # Computing current width
        curr_width = math.floor(width_coef * i + first_layer_width)

        # Creating current layer
        network = fully_connected(network, curr_width, activation=hidden_act, name='fc' + str(i), weights_init=winit,
                                  weight_decay=weight_decay)

        print("size : " + str(curr_width))

        # Applying dropout
        network = dropout(network, dropout_val)

    # Adding outlayer
    network = fully_connected(network, 1, activation=outlayer_act, name='outlayer', weights_init=winit)

    # Adam optimizer creation
    adam = Adam(learning_rate=learning_rate, epsilon=epsilon)

    # Model evaluation layer creation
    network = regression(network, optimizer=adam,
                         loss=cost_fun, metric=validation_fun, name='target')

    return network


def train_model(input_X_h5_loc, labels_y_h5_loc, model_name, model_loc, logs_loc, epochs, last_layer_width,
                samples_per_batch=1000, learning_rate=0.001, epsilon=1e-8, dropout=0.99, stddev_init=0.001,
                hidden_act='relu', outlayer_act='prelu', depth=2,
                weight_decay=0.001, gpu_mem_prop=1, save_model=True):
    """
    Trains a neural network with the given data and the given parameters
    :param input_X_h5_loc:
    :param labels_y_h5_loc:
    :param model_name:
    :param model_loc:
    :param epochs:
    :param last_layer_width:
    :param samples_per_batch:
    :param learning_rate:
    :param epsilon:
    :param dropout:
    :param stddev_init:
    :param hidden_act:
    :param outlayer_act:
    :param depth:
    :param weight_decay:
    :param gpu_mem_prop:
    :param save_model:
    :return:
    """

    tf.reset_default_graph()

    # Loading inputs and targets
    input_X = np.array(h5py.File(input_X_h5_loc)[inputs_key])
    labels_y = np.array(h5py.File(labels_y_h5_loc)[targets_key])

    # Computing first layer width (all the examples of the dataset must have the same width)
    first_layer_width = len(input_X[0])

    print("First layer width : "+str(first_layer_width))

    # Creating NN
    network = _nn_creation(first_layer_width, last_layer_width, depth, epsilon, learning_rate, dropout, stddev_init,
                          hidden_act, outlayer_act, weight_decay, _rmse_valid, _rmse, gpu_mem_prop)

    # Model creation
    model = tfl.DNN(network, tensorboard_verbose=3, tensorboard_dir=logs_loc)

    # Training
    model.fit(X_inputs=input_X, Y_targets=labels_y, batch_size=samples_per_batch,
              shuffle=True, snapshot_step=100, validation_set=0.1,
              show_metric=True, run_id=model_name, n_epoch=epochs)

    # Saving model
    if save_model:
        model.save(model_loc)


def predict(model_loc, test_prepared_input_loc, test_labels_loc, batch_size, last_layer_width,
            depth, hidden_act, outlayer_act):
    """
    Predicts given test data with the given neural network and returns the error vector and the prediction vector
    :param model_loc:
    :param test_prepared_input_loc:
    :param test_labels_loc:
    :param batch_size:
    :param last_layer_width:
    :param depth:
    :return: error vector, prediction vector, targets vector
    """
    # Loading inputs and targets
    input_X = np.array(h5py.File(test_prepared_input_loc)[inputs_key])
    labels_y = np.array(h5py.File(test_labels_loc)[targets_key])

    # Computing first layer width (all the examples of the dataset must have the same width)
    first_layer_width = len(input_X[0])

    # NN Creation
    network = _nn_creation(first_layer_width, last_layer_width, depth, hidden_act=hidden_act, outlayer_act=outlayer_act)

    # Model creation
    model = tfl.DNN(network)

    # Loading weights
    model.load(model_loc, weights_only=True)

    # Prediction
    i = 0
    predictions = []
    while i < len(input_X):
        j = min(len(input_X), i + batch_size)
        predictions.extend(model.predict(np.array(input_X[i:j])))
        i += batch_size

    return _rmse_test(labels_y, predictions).reshape(1, -1)[0], predictions, labels_y
