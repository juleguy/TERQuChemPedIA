import json
from json_keys import *
from data_preparation import data_split
from data_preparation import generate_data_wished_size, generate_data
import model_nn
import model_svm
import stats
import bonds_lengths_stats
import paths
from grid_search import grid_search_cv
import sys
import models
import model_kernel_ridge
import model_ridge


def _load_paths(paths_json):
    """
    Loads current paths in memory
    :param paths_json: json object containing the paths
    :return: None
    """
    if input_data_loc_k in paths_json:
        paths.input_data_loc = paths_json[input_data_loc_k]

    if train_set_loc_k in paths_json:
        paths.train_set_loc = paths_json[train_set_loc_k]

    if test_set_loc_k in paths_json:
        paths.test_set_loc = paths_json[test_set_loc_k]

    if train_prepared_input_loc_k in paths_json:
        paths.train_prepared_input_loc = paths_json[train_prepared_input_loc_k]

    if test_prepared_input_loc_k in paths_json:
        paths.test_prepared_input_loc = paths_json[test_prepared_input_loc_k]

    if train_labels_loc_k in paths_json:
        paths.train_labels_loc = paths_json[train_labels_loc_k]

    if test_labels_loc_k in paths_json:
        paths.test_labels_loc = paths_json[test_labels_loc_k]

    if bonds_lengths_loc_k in paths_json:
        paths.bonds_lengths_loc = paths_json[bonds_lengths_loc_k]

    if plots_dir_k in paths_json:
        paths.plots_dir = paths_json[plots_dir_k]

    if model_loc_k in paths_json:
        paths.model_loc = paths_json[model_loc_k]

    if dataset_loc_k in paths_json:
        paths.dataset_loc = paths_json[dataset_loc_k]

    if logs_dir_k in paths_json:
        paths.logs_dir = paths_json[logs_dir_k]

    if models_dir_k in paths_json:
        paths.models_dir = paths_json[models_dir_k]


def _check_key(container_json, key):
    if key not in container_json:
        raise RuntimeError(str(container_json)+" must contain "+str(key)+" object.")


def _data_split(data_split_json):
    """
    Splits a dataset into a train set and a test set
    :param data_split_json: json object containing the instructions
    :return:
    """

    # Loading paths if specified
    if paths_k in data_split_json:
        _load_paths(data_split_json[paths_k])

    # Checking keys
    _check_key(data_split_json, params_k)
    params = data_split_json[params_k]
    _check_key(params, train_proportion_k)
    _check_key(params, random_state_k)

    # Extracting values
    train_proportion = float(params[train_proportion_k])
    random_state = int(params[random_state_k])

    if paths.input_data_loc == "":
        raise RuntimeError(input_data_loc_k + " cannot be empty")

    if paths.train_set_loc == "":
        raise RuntimeError(train_set_loc_k + " cannot be empty")

    # Dataset splitting
    data_split(paths.input_data_loc, paths.train_set_loc, paths.test_set_loc, train_proportion, random_state)


def _data_preparation(prepare_data_json):
    """
    Prepares inputs and targets for a model on test and train set.
    Two options are available :
    * generation of train and test datasets that approximate given sizes : use of cross multiplication
    * generation of train and test datasets from an exact number of molecules
    :param prepare_data_json:
    :return:
    """
    # Loading paths if specified
    if paths_k in prepare_data_json:
        _load_paths(prepare_data_json[paths_k])

    # Checking keys and loading values
    _check_key(prepare_data_json, selected_mols_k)
    _check_key(prepare_data_json, params_k)

    selected_mols_json = prepare_data_json[selected_mols_k]
    _check_key(selected_mols_json, mol_min_size_k)
    _check_key(selected_mols_json, mol_max_size_k)
    _check_key(selected_mols_json, max_anum_k)
    _check_key(selected_mols_json, anum_1_k)
    _check_key(selected_mols_json, anum_2_k)
    _check_key(selected_mols_json, min_bond_size_k)
    _check_key(selected_mols_json, max_bond_size_k)
    _check_key(selected_mols_json, bond_max_neighbours_k)

    mol_min_size = int(selected_mols_json[mol_min_size_k])
    mol_max_size = int(selected_mols_json[mol_max_size_k])
    max_anum = int(selected_mols_json[max_anum_k])
    anum_1 = int(selected_mols_json[anum_1_k])
    anum_2 = int(selected_mols_json[anum_2_k])
    min_bond_size = float(selected_mols_json[min_bond_size_k])
    max_bond_size = float(selected_mols_json[max_bond_size_k])
    bond_max_neighbours = int(selected_mols_json[bond_max_neighbours_k])

    params_json = prepare_data_json[params_k]

    _check_key(params_json, pos_class_k)
    _check_key(params_json, one_hot_anums_k)
    _check_key(params_json, amasses_k)
    _check_key(params_json, distances_k)
    _check_key(params_json, distances_cut_off_k)
    _check_key(params_json, batch_size_k)

    pos_class = params_json[pos_class_k] == "True"
    one_hot_anums = params_json[one_hot_anums_k] == "True"
    amasses = params_json[amasses_k] == "True"
    if params_json[distances_cut_off_k] == "None":
        distances_cut_off = None
    else:
        distances_cut_off = float(params_json[distances_cut_off_k])
    distances = params_json[distances_k] == "True"
    batch_size = int(params_json[batch_size_k])

    # Checking distances_fun in distances specified and loading value
    if distances:
        _check_key(params_json, distances_fun_k)
        distances_fun_str = params_json[distances_fun_k]

    # Checking that paths are loaded
    if paths.train_set_loc == "":
        raise RuntimeError(train_set_loc_k + " cannot be empty")

    if paths.train_prepared_input_loc == "":
        raise RuntimeError(train_prepared_input_loc_k + " cannot be empty")

    if paths.train_labels_loc == "":
        raise RuntimeError(train_labels_loc_k + " cannot be empty")

    # Wished size mode
    if wished_train_size_k in params_json and wished_test_size_k in params_json:

        wished_train_size = int(params_json[wished_train_size_k])
        wished_test_size = int(params_json[wished_test_size_k])

        # Generating data on train set
        generate_data_wished_size(paths.train_set_loc, paths.train_prepared_input_loc, paths.train_labels_loc, anum_1,
                                  anum_2, wished_train_size, batch_size, max_anum, min_bond_size, max_bond_size,
                                  mol_min_size, mol_max_size, distances_cut_off, one_hot_anums, distances, pos_class,
                                  amasses, distances_fun_str, bond_max_neighbours)

        # Generating data on test set
        generate_data_wished_size(paths.test_set_loc, paths.test_prepared_input_loc, paths.test_labels_loc, anum_1,
                                  anum_2, wished_test_size, batch_size, max_anum, min_bond_size, max_bond_size,
                                  mol_min_size, mol_max_size, distances_cut_off, one_hot_anums, distances, pos_class,
                                  amasses, distances_fun_str, bond_max_neighbours)

    # Number of molecules mode
    elif nb_mol_from_train_k in params_json and nb_mol_from_test_k in params_json:

        nb_mol_from_train = int(params_json[nb_mol_from_train_k])
        nb_mol_from_test = int(params_json[nb_mol_from_test_k])

        # Generating data on train set
        generate_data(paths.train_set_loc, paths.train_prepared_input_loc, paths.train_labels_loc, anum_1, anum_2,
                      nb_mol_from_train, batch_size, max_anum, min_bond_size, max_bond_size, mol_min_size, mol_max_size,
                      distances_cut_off, one_hot_anums, distances, pos_class, amasses, distances_fun_str,
                      bond_max_neighbours)

        # Generating data on test set
        generate_data(paths.test_set_loc, paths.test_prepared_input_loc, paths.test_labels_loc, anum_1, anum_2,
                      nb_mol_from_test, batch_size, max_anum, min_bond_size, max_bond_size, mol_min_size, mol_max_size,
                      distances_cut_off, one_hot_anums, distances, pos_class, amasses, distances_fun_str,
                      bond_max_neighbours)

    else:
        raise RuntimeError("A couple ("+wished_test_size_k+", "+wished_test_size_k+") or ("+nb_mol_from_test_k+", " +
                           nb_mol_from_train_k+") must be specified")


def _model_train(model_train_json):
    """
    Trains a model
    :param model_train_json:
    :return:
    """

    # Loading paths if specified
    if paths_k in model_train_json:
        _load_paths(model_train_json[paths_k])

    # Checking keys and loading values
    _check_key(model_train_json, model_name_k)
    model_name = model_train_json[model_name_k]
    _check_key(model_train_json, model_type_k)
    model_type = model_train_json[model_type_k]
    _check_key(model_train_json, params_k)
    params_json = model_train_json[params_k]
    _check_key(params_json, save_model_k)
    save_model = params_json[save_model_k] == "True"

    # Checking that paths are specified
    if paths.train_prepared_input_loc == "":
        raise RuntimeError(train_prepared_input_loc_k + " cannot be empty")

    if paths.train_labels_loc == "":
        raise RuntimeError(train_set_loc_k + " cannot be empty")

    if save_model and paths.model_loc == "":
        raise RuntimeError(model_loc_k + " cannot be empty since the model must be saved")

    # Training neural network
    if model_type == NN_k:

        # Checking presence of neural network parameters
        _check_key(params_json, epochs_k)
        _check_key(params_json, last_layer_width_k)
        _check_key(params_json, batch_size_k)
        _check_key(params_json, learning_rate_k)
        _check_key(params_json, epsilon_k)
        _check_key(params_json, dropout_k)
        _check_key(params_json, stddev_init_k)
        _check_key(params_json, hidden_act_k)
        _check_key(params_json, outlayer_act_k)
        _check_key(params_json, depth_k)
        _check_key(params_json, weight_decay_k)
        _check_key(params_json, gpu_mem_prop_k)

        # Loading parameters
        epochs = int(params_json[epochs_k])
        last_layer_width = int(params_json[last_layer_width_k])
        batch_size = int(params_json[batch_size_k])
        learning_rate = float(params_json[learning_rate_k])
        epsilon = float(params_json[epsilon_k])
        dropout = float(params_json[dropout_k])
        stddev_init = float(params_json[stddev_init_k])
        hidden_act = params_json[hidden_act_k]
        outlayer_act = params_json[outlayer_act_k]
        depth = int(params_json[depth_k])
        weight_decay = float(params_json[weight_decay_k])
        gpu_mem_prop = float(params_json[gpu_mem_prop_k])

        # Checking that a log path has been specified
        if paths.logs_dir == "":
            raise RuntimeError(logs_dir_k + " logs_loc cannot be empty")

        # Training the model
        model_nn.train_model(paths.train_prepared_input_loc, paths.train_labels_loc, model_name, paths.model_loc,
                             paths.logs_dir, epochs, last_layer_width, batch_size, learning_rate, epsilon, dropout,
                             stddev_init, hidden_act, outlayer_act, depth, weight_decay, gpu_mem_prop, save_model)

    # Training SVM model
    elif model_type == SVM_k:

        # Checking presence of SVM parameters
        _check_key(params_json, kernel_k)
        _check_key(params_json, C_k)
        _check_key(params_json, epsilon_k)
        _check_key(params_json, coef0_k)
        _check_key(params_json, shrinking_k)
        _check_key(params_json, tol_k)
        _check_key(params_json, cache_size_k)
        _check_key(params_json, verbose_k)
        _check_key(params_json, max_iter_k)

        # Loading kernel type and checking and loading specific kernel parameters presence
        kernel = params_json[kernel_k]

        if kernel == "rbf" or kernel == "sigmoid" or kernel == "poly":
            _check_key(params_json, gamma_k)
            if params_json[gamma_k] == "auto":
                gamma = "auto"
            else:
                gamma = float(params_json[gamma_k])
        else:
            gamma = 0.1

        if kernel == "poly":
            _check_key(params_json, degree_k)
            degree = int(params_json[degree_k])
        else:
            degree = 1

        # Loading generic SVM parameters
        epsilon = float(params_json[epsilon_k])
        coef0 = float(params_json[coef0_k])
        shrinking = params_json[shrinking_k] == "True"
        tol = float(params_json[tol_k])
        cache_size = float(params_json[cache_size_k])
        verbose = params_json[verbose_k] == "True"
        max_iter = int(params_json[max_iter_k])
        C = float(params_json[C_k])

        # Training the model
        model_svm.train_model(paths.train_prepared_input_loc, paths.train_labels_loc, paths.model_loc, C, kernel,
                              epsilon, degree, gamma, coef0, shrinking, tol, cache_size, verbose, max_iter, save_model)

    # Training kernel ridge model
    elif model_type == kernel_ridge_k:

        # Checking presence of kernel ridge parameters
        _check_key(params_json, kernel_k)
        _check_key(params_json, alpha_k)
        _check_key(params_json, coef0_k)

        # Loading kernel type and checking and loading specific kernel parameters presence
        kernel = params_json[kernel_k]

        if kernel == "poly":
            _check_key(params_json, gamma_k)
            if params_json[gamma_k] == "auto" or params_json[gamma_k] == "None":
                gamma = None
            else:
                gamma = float(params_json[gamma_k])

            _check_key(params_json, degree_k)
            degree = int(params_json[degree_k])
        else:
            gamma = None
            degree = 1

        # Loading generic kernel ridge parameters
        alpha = float(params_json[alpha_k])
        coef0 = float(params_json[coef0_k])

        # Training the model
        model_kernel_ridge.train_model(paths.train_prepared_input_loc, paths.train_labels_loc, paths.model_loc, alpha,
                                       kernel, gamma, degree, coef0, save_model)

    # Training ridge regression model
    elif model_type == ridge_k:

        # Checking presence of ridge parameters
        _check_key(params_json, alpha_k)
        _check_key(params_json, max_iter_k)
        _check_key(params_json, tol_k)

        # Loading ridge parameters
        alpha = float(params_json[alpha_k])
        max_iter = int(params_json[max_iter_k])
        tol = float(params_json[tol_k])

        # Training the model
        model_ridge.train_model(paths.train_prepared_input_loc, paths.train_labels_loc, paths.model_loc, alpha,
                                max_iter, tol, save_model)


def _bonds_stats(bonds_stats_json):
    """
    Records bonds lengths of pairs of atoms
    :param bonds_stats_json:
    :return:
    """

    # Loading paths if specified
    if paths_k in bonds_stats_json:
        _load_paths(bonds_stats_json[paths_k])

    # Checking keys and loading values
    _check_key(bonds_stats_json, params_k)
    params = bonds_stats_json[params_k]

    # Checking that the json is complete
    _check_key(params, max_anum_k)
    _check_key(params, mol_min_size_k)
    _check_key(params, mol_max_size_k)
    _check_key(params, anum_1_k)
    _check_key(params, anum_2_k)
    _check_key(params, batch_size_k)

    # Loading parameters
    max_anum = int(params[max_anum_k])
    mol_max_size = int(params[mol_max_size_k])
    mol_min_size = int(params[mol_min_size_k])
    anum_1 = int(params[anum_1_k])
    anum_2 = int(params[anum_2_k])
    batch_size = int(params[batch_size_k])

    # Checking that paths are specified
    if paths.dataset_loc == "":
        raise RuntimeError(dataset_loc_k + " cannot be empty")

    if paths.bonds_lengths_loc == "":
        raise RuntimeError(bonds_lengths_loc_k + " cannot be empty")

    # Computing and saving the bonds lengths
    bonds_lengths_stats.record_bonds_lengths(paths.dataset_loc, paths.bonds_lengths_loc, anum_1, anum_2, batch_size,
                                             max_anum, mol_min_size, mol_max_size)


def _predict(model_type, params):
    """ Prediction on the test set """

    _check_key(params, batch_size_k)
    batch_size = int(params[batch_size_k])

    # Predictions of a neural network model
    if model_type == NN_k:
        # Checking that specific attributes for NN are specified
        _check_key(params, last_layer_width_k)
        _check_key(params, depth_k)
        _check_key(params, hidden_act_k)
        _check_key(params, outlayer_act_k)

        last_layer_width = int(params[last_layer_width_k])
        depth = int(params[depth_k])
        hidden_act = params[hidden_act_k]
        outlayer_act = params[outlayer_act_k]

        return model_nn.predict(paths.model_loc, paths.test_prepared_input_loc, paths.test_labels_loc, batch_size,
                                last_layer_width, depth, hidden_act, outlayer_act)

    elif model_type == SVM_k or model_type == kernel_ridge_k or model_type == ridge_k:
        return models.predict(paths.model_loc, paths.test_prepared_input_loc, paths.test_labels_loc, batch_size)


def _plot_predictions(plot_predictions_json):
    """
    Plots the result of the predictions of a model
    :param plot_predictions_json:
    :return:
    """

    # Loading paths if specified
    if paths_k in plot_predictions_json:
        _load_paths(plot_predictions_json[paths_k])

    # Checking keys and loading values
    _check_key(plot_predictions_json, params_k)
    params = plot_predictions_json[params_k]

    # Checking that the json is complete
    _check_key(params, model_name_k)
    _check_key(params, anum_1_k)
    _check_key(params, anum_2_k)
    _check_key(params, plot_error_distrib_k)
    _check_key(params, plot_targets_error_distrib_k)
    _check_key(params, plot_targets_predictions_k)
    _check_key(params, model_type_k)
    _check_key(params, display_plots_k)

    # Loading parameters
    model_name = params[model_name_k]
    anum_1 = int(params[anum_1_k])
    anum_2 = int(params[anum_2_k])
    plot_error_distrib = params[plot_error_distrib_k] == "True"
    plot_targets_error_distrib = params[plot_targets_error_distrib_k] == "True"
    plot_targets_predictions = params[plot_targets_predictions_k] == "True"
    display_plots = params[display_plots_k] == "True"
    model_type = params[model_type_k]

    # Checking that paths are specified
    if paths.model_loc == "":
        raise RuntimeError(model_loc_k + " cannot be empty")
    if paths.test_prepared_input_loc == "":
        raise RuntimeError(test_prepared_input_loc_k + " cannot be empty")
    if paths.test_labels_loc == "":
        raise RuntimeError(test_labels_loc_k + " cannot be empty")
    if paths.plots_dir == "":
        raise RuntimeError(plots_dir_k + " cannot be empty")
    if paths.bonds_lengths_loc == "":
        raise RuntimeError(bonds_lengths_loc_k + " cannot be empty")

    # Prediction
    errors, predictions, targets = _predict(model_type, params)

    # Plot
    stats.plot_model_results(errors, predictions, targets, model_name, anum_1, anum_2, paths.bonds_lengths_loc,
                             paths.plots_dir, plot_error_distrib, plot_targets_error_distrib, plot_targets_predictions,
                             display_plots)


def _check_grid(params, grid_params):
    """
    Checks if all of the specified params are present in the specified grid
    :param params: list of required parameters
    :param grid_params: grid to give to scikit learn GridSearchCV
    :return:
    """
    for nn_param in params:
        for grid in grid_params:
            if nn_param not in grid:
                raise RuntimeError(nn_param + " must be specified in grid " + str(grid))


def _string_list_to_bool_list(string_list):
    """
    Converts a list of strings representing booleans ("True", "False") into a boolean list of two elements at most
    :param string_list:
    :return:
    """

    boolean_list = []

    if "True" in string_list:
        boolean_list.append(True)

    if "False" in string_list:
        boolean_list.append(False)

    return boolean_list


def _grid_search_cv(grid_search_json):
    """
    Grid searching best parameters for a model
    :param grid_search_json:
    :return:
    """

    # Loading paths if specified
    if paths_k in grid_search_json:
        _load_paths(grid_search_json[paths_k])

    # Checking keys and loading grid_params object
    _check_key(grid_search_json, grid_params_k)
    grid_params = grid_search_json[grid_params_k]

    # Checking keys and loading params object
    _check_key(grid_search_json, params_k)
    params = grid_search_json[params_k]

    # Checking that the param object is complete
    _check_key(params, model_type_k)
    _check_key(params, n_jobs_k)
    _check_key(params, cv_k)

    # Loading params
    model_type = params[model_type_k]
    n_jobs = int(params[n_jobs_k])
    cv = int(params[cv_k])

    # Checking that the prepared input and the targets have been specified
    if paths.train_prepared_input_loc == "":
        raise RuntimeError(train_prepared_input_loc_k + " cannot be empty")

    if paths.train_labels_loc == "":
        raise RuntimeError(train_set_loc_k + " cannot be empty")

    # Checking specific parameters
    if model_type == NN_k:

        # Loading NN specific param
        _check_key(params, gpu_mem_prop_k)
        gpu_mem_prop = float(params[gpu_mem_prop_k])

        # Adding NN specific param to the grids
        for grid in grid_params:
            grid[gpu_mem_prop_k] = [gpu_mem_prop]
            grid[logs_dir_k] = [paths.logs_dir]

        # Checking that all the grids are complete
        nn_params = [learning_rate_k, epsilon_k, dropout_k, stddev_init_k, hidden_act_k, outlayer_act_k, weight_decay_k,
                     last_layer_width_k, depth_k, batch_size_k, epochs_k]
        _check_grid(nn_params, grid_params)

    elif model_type == SVM_k:

        # Checking that all the grids are complete
        svm_params = [kernel_k, epsilon_k, coef0_k, shrinking_k, tol_k, cache_size_k, verbose_k, max_iter_k, C_k,
                      degree_k, gamma_k]
        _check_grid(svm_params, grid_params)

        # Converting "True" and "False" strings to boolean values
        boolean_attributes = [shrinking_k, verbose_k]
        for grid in grid_params:
            for boolean_attribute in boolean_attributes:
                grid[boolean_attribute] = _string_list_to_bool_list(grid[boolean_attribute])

    elif model_type == ridge_k:

        # Checking that all the grids are complete
        ridge_params = [alpha_k, max_iter_k, tol_k, verbose_k]
        _check_grid(ridge_params, grid_params)

    elif model_type == kernel_ridge_k:

        # Checking that all the grids are complete
        ridge_params = [alpha_k, kernel_k, gamma_k, degree_k, coef0_k]
        _check_grid(ridge_params, grid_params)

    # Grid search
    grid_search_cv(model_type, paths.train_prepared_input_loc, paths.train_labels_loc, grid_params, cv, n_jobs)


def execute(json_path):

    # Opening json file
    with open(json_path) as json_data:
        data = json.load(json_data)

        # Iterating over all the tasks
        for task in data["tasks"]:

            # Loading paths
            if paths_k in data:
                _load_paths(data[paths_k])

            # Executing potential data split
            if data_split_k in task:
                _data_split(task[data_split_k])

            # Executing potential model data preparation
            elif prepare_model_data_k in task:
                _data_preparation(task[prepare_model_data_k])

            # Executing potential model training
            elif model_train_k in task:
                _model_train(task[model_train_k])

            # Executing potential bonds lengths recording
            elif bonds_stats_k in task:
                _bonds_stats(task[bonds_stats_k])

            # Executing potential model results plotting
            elif plot_predictions_k in task:
                _plot_predictions(task[plot_predictions_k])

            # Executing potential grid search
            elif grid_search_cv_k in task:
                _grid_search_cv(task[grid_search_cv_k])


# Executing given json files
for arg in sys.argv[1:]:
    print(sys.argv)
    execute(arg)
