from sklearn.svm import SVR
import h5py
from h5_keys import *
import numpy as np
from sklearn.externals import joblib
import os
import time


def train_model(input_X_h5_loc, labels_y_h5_loc, model_loc, C, kernel, epsilon, degree, gamma, coef0,
                shrinking, tol, cache_size, verbose, max_iter, save_model):
    """
    Trains a SVM regressor with to the given parameters

    See Scikit-learn documentation : http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

    :param input_X_h5_loc: input data location
    :param labels_y_h5_loc: labels location
    :param model_loc: path to the location the model will be saved at
    :param C: penalty parameter on the error term (default 1.0)
    :param kernel: kernel type ('rbf', 'linear', 'poly', 'sigmoid')
    :param epsilon: epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty
    is associated in the training loss function with points predicted within a distance epsilon from the actual value.
    :param degree: degree of the polynomial kernel
    :param gamma: kernel coefficient for rbf, poly and sigmoid
    :param coef0: independant term in kernel function (default 0.0)
    :param shrinking: whether to use the shrinking heuristic
    :param tol: tolerance for stopping criterion
    :param cache_size: size of the kernel cache (in MB, default : 200)
    :param verbose: enable or disable verbose output
    :param max_iter: hard limit on iterations within solver, or -1 for no limit.
    :param save_model: whether the model must me saved
    :return:
    """

    total_time = time.time()

    # Loading inputs and targets
    input_X = np.array(h5py.File(input_X_h5_loc)[inputs_key])
    labels_y = np.array(h5py.File(labels_y_h5_loc)[targets_key]).reshape((-1,))

    # Creating model
    model = SVR(kernel, degree, gamma, coef0, tol, C, epsilon, shrinking, cache_size, verbose, max_iter)

    # Model training
    model.fit(input_X, labels_y)

    # Saving the model if specified
    if save_model:
        os.makedirs(model_loc[:model_loc.rindex(os.path.sep)], exist_ok=True)
        joblib.dump(model, model_loc)

    print("--- %s seconds ---" % (time.time() - total_time))
