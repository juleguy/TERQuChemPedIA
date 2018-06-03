import uuid
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.optimizers import Adam
import math
import h5py
import numpy as np
from h5_keys import *
from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
import tflearn as tfl
from sklearn.model_selection import GridSearchCV


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
    width_coef = (last_layer_width - first_layer_width) / (depth + 1)

    # Creating hidden layers
    for i in range(depth):

        # Computing current width
        curr_width = math.floor(width_coef * (i+1) + first_layer_width)

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


class NNRegressor(BaseEstimator, RegressorMixin):
    """ Wrapper around TFLearn to use models as Scikit-learn models
    Inspired of http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/ """

    def __init__(self, logs_dir=None, models_dir=None, learning_rate=None, epsilon=None, dropout=None,
                 stddev_init=None, hidden_act=None, outlayer_act=None, weight_decay=None,
                 last_layer_width=None, depth=None, batch_size=None, epochs=None, gpu_mem_prop=None, save_model=None,
                 score_fun=_rmse_valid, loss_fun=_rmse):

        self.logs_dir = logs_dir
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.dropout = dropout
        self.stddev_init = stddev_init
        self.hidden_act = hidden_act
        self.outlayer_act = outlayer_act
        self.weight_decay = weight_decay
        self.last_layer_width = last_layer_width
        self.depth = depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fun = loss_fun
        self.score_fun = score_fun
        self.gpu_mem_prop = gpu_mem_prop
        self.save_model = save_model
        self.models_dir = models_dir
        self.model = None

    def fit(self, X, y=None):
        tf.reset_default_graph()
        tfl.init_graph(gpu_memory_fraction=self.gpu_mem_prop)

        # Computing model name according to its parameters so that it can be easily retrieved if saved.
        # Also adding an ID of 8 characters, so that the logs are recorded in different directories
        # There is no guarantee that the ID is unique but collisions should not happen often and that would only
        # mess with tensorboard graphs
        model_name = ("lr" + str(self.learning_rate) + "|eps" + str(self.epsilon) + "|do" + str(self.dropout) +
                      "|stddev_init" + str(self.stddev_init) + "|hidact" + self.hidden_act + "|out" +
                      self.outlayer_act + "|wd" + str(self.weight_decay) + "|lastlayw" + str(self.last_layer_width) +
                      "|d" + str(self.depth) + "|batchs" + str(self.batch_size) + "|epochs" + str(self.epochs) +
                      "|id"+str(uuid.uuid4())[:8])

        # Computing first layer width (all the examples of the dataset must have the same width)
        first_layer_width = len(X[0])

        # Neural network creation
        network = _nn_creation(first_layer_width, self.last_layer_width, self.depth, self.epsilon, self.learning_rate,
                               self.dropout, self.stddev_init, self.hidden_act, self.outlayer_act, self.weight_decay,
                               self.score_fun, self.loss_fun, self.gpu_mem_prop)

        # Model creation
        self.model = tfl.DNN(network, tensorboard_verbose=3, tensorboard_dir=self.logs_dir)

        # Training
        self.model.fit(X_inputs=X, Y_targets=y, batch_size=self.batch_size,
                       shuffle=True, snapshot_step=100, validation_set=0.1,
                       show_metric=True, run_id=model_name, n_epoch=self.epochs)

        if self.save_model:
            self.model.save(self.models_dir+model_name+".tflearn")

    def predict(self, X, y=None):
        return self.model.predict(X, y)

    def score(self, X, y=None):
        return self.model.evaluate(X, y, batch_size=100)[0]


def grid_search_cv(train_prepared_input_loc, train_labels_loc, parameters_grid, cv, n_jobs):
    """
    Performs a grid search of the specified parameters on the specified set.
    Uses scikit-learn GridSearchCV method that also performs a cross validation.
    :param train_prepared_input_loc: inputs location
    :param train_labels_loc: targets location
    :param parameters_grid: grid of parameters
    :param cv: number of cross validations
    :param n_jobs: number of cpu jobs
    :return:
    """

    # Loading inputs and targets
    input_X = np.array(h5py.File(train_prepared_input_loc)[inputs_key])
    labels_y = np.array(h5py.File(train_labels_loc)[targets_key])

    # Creating the Scikit-Learn model from our Scikit-Learn like regressor
    grid_search = GridSearchCV(estimator=NNRegressor(), param_grid=parameters_grid, cv=cv, n_jobs=n_jobs)

    # Grid search
    grid_search.fit(input_X, labels_y)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    print(grid_search.best_estimator_.get_params())
