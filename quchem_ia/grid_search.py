import numpy as np
import h5py
from h5_keys import *
from model_nn import NNRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from json_keys import *
from sklearn.kernel_ridge import KernelRidge


def grid_search_cv(model_type, train_prepared_input_loc, train_labels_loc, parameters_grid, cv, n_jobs):
    """
    Performs a grid search of the specified parameters on the specified set.
    Uses scikit-learn GridSearchCV method that also performs a cross validation.

    Part of the code extracted from :
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

    :param model_type : type of the model
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

    # Model creation
    if model_type == NN_k:
        # Creating the Scikit-Learn model from our Scikit-Learn like regressor
        model = NNRegressor()

    else:
        labels_y = labels_y.reshape((-1,))

        if model_type == SVM_k:
            model = SVR()

        elif model_type == ridge_k:
            model = Ridge()

        elif model_type == kernel_ridge_k:
            model = KernelRidge()

    grid_search = GridSearchCV(estimator=model, param_grid=parameters_grid, cv=cv, n_jobs=n_jobs)

    # Grid search
    grid_search.fit(input_X, labels_y)

    print()
    print("Grid scores on development set:")
    print()
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    print(grid_search.best_estimator_.get_params())
