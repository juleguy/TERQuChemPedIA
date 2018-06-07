import h5py
from h5_keys import *
import numpy as np
from sklearn.externals import joblib
import os
import time
from sklearn.kernel_ridge import KernelRidge


def train_model(input_X_h5_loc, labels_y_h5_loc, model_loc, alpha, kernel, gamma, degree, coef0, save_model):
    """
    Trains a kernel ridge regression model

    See Scikit-learn documentation : http://scikit-learn.org/stable/modules/generated/sklearn.
                                            kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge
    """

    total_time = time.time()

    # Loading inputs and targets
    input_X = np.array(h5py.File(input_X_h5_loc)[inputs_key])
    labels_y = np.array(h5py.File(labels_y_h5_loc)[targets_key]).reshape((-1,))

    # Creating model
    model = KernelRidge(degree=degree, coef0=coef0, kernel=kernel, gamma=gamma, alpha=alpha)

    # Model training
    model.fit(input_X, labels_y)

    # Saving the model if specified
    if save_model:
        os.makedirs(model_loc[:model_loc.rindex(os.path.sep)], exist_ok=True)
        joblib.dump(model, model_loc)

    print("--- %s seconds ---" % (time.time() - total_time))



