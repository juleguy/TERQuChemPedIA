import numpy as np
import h5py
from sklearn.externals import joblib
from h5_keys import *


def _rmse_test(targets, predictions):
    """
    Computes the absolute error on the prediction on each example
    :param targets: targets
    :param predictions: predictions
    :return: absolute error vector
    """
    return np.sqrt(np.square(np.diff([targets, predictions], axis=0)))[0]


def predict(model_loc, test_prepared_input_loc, test_labels_loc, batch_size):
    """
    Predicts given test data with the given scikit learn model and returns the error vector, the prediction vector
    and the targets vector
    :param model_loc : model location
    :param test_prepared_input_loc : test prepared input location
    :param test_labels_loc : test labels location
    :param batch_size : number of predicted values in a single batch
    :return errors, predicions and labels (in pm)
    """

    # Loading inputs and targets
    input_X = np.array(h5py.File(test_prepared_input_loc)[inputs_key])
    labels_y = np.array(h5py.File(test_labels_loc)[targets_key]).reshape((-1,))

    # Loading model
    model = joblib.load(model_loc)

    # Prediction
    i = 0
    predictions = []
    while i < len(input_X):
        j = min(len(input_X), i + batch_size)
        predictions.extend(model.predict(np.array(input_X[i:j])))
        i += batch_size

    predictions = np.array(predictions)

    return (_rmse_test(labels_y, predictions).reshape(1, -1)[0])/10, predictions/10, labels_y/10
