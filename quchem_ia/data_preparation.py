import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from h5_keys import *
import time
import math


def data_split(input_data_loc, train_set_loc, test_set_loc, train_proportion=0.9, random_state=12):
    """
    Splitting a dataset into a train set and a test set using scikit-learn

    :param input_data_loc: Original dataset location
    :param train_set_loc: New train set location
    :param test_set_loc: New test set location
    :param train_proportion: Proportion of data put in the train dataset (default=0.9)
    :param random_state: Random state used to split the set
    :return: None
    """

    # Loading the original dataset (read-only mode)
    original_dataset = h5py.File(input_data_loc, 'r')

    # Loading the complete original set into a numpy array
    print("Loading the data...")
    data = np.array([np.array(original_dataset[pubchem_id_key], dtype=np.int32),
                     np.array(original_dataset[anums_key]),
                     np.array(original_dataset[amasses_key]),
                     np.array(original_dataset[coords_key])]).T

    # Calling scikit-learn to split the data
    print("Separation of the data...")
    train_set, test_set = train_test_split(data, train_size=train_proportion, random_state=random_state)

    # Transposing back matrices
    train_set = train_set.T
    test_set = test_set.T

    # Creating new split datasets h5 files
    print("Creating h5 files...")
    train_set_h5 = h5py.File(train_set_loc, 'w')
    test_set_h5 = h5py.File(test_set_loc, 'w')

    try:
        # Float arrays h5py special type
        varlen_floatarray = h5py.special_dtype(vlen=np.dtype("float32"))

        # Computing output sizes
        train_size = len(train_set[0])
        test_size = len(test_set[0])

        print("train size = " + str(train_size))

        col_names = [pubchem_id_key, anums_key, amasses_key, coords_key]
        datatypes = [np.int32, varlen_floatarray, varlen_floatarray, varlen_floatarray]

        # Creating four datasets for each output file
        print("Creating datasets...")

        # Creating pubchem ids datasets
        train_set_h5.create_dataset(col_names[0], data=np.array(train_set[0], dtype=np.int32), shape=(train_size,),
                                    dtype=datatypes[0], compression="gzip", chunks=True)

        test_set_h5.create_dataset(col_names[0], data=np.array(test_set[0], dtype=np.int32), shape=(test_size,),
                                   dtype=datatypes[0], compression="gzip", chunks=True)

        # Creating anums, amasses and coords datasets
        for i in range(1, 4):
            train_set_h5.create_dataset(col_names[i], data=train_set[i], shape=(train_size,),
                                        dtype=datatypes[i], compression="gzip", chunks=True)

            test_set_h5.create_dataset(col_names[i], data=test_set[i], shape=(test_size,),
                                       dtype=datatypes[i], compression="gzip", chunks=True)

        # Writing data to disk
        print("Writing train set to disk...")
        train_set_h5.flush()

        print("Writing test set to disk...")
        test_set_h5.flush()

        print("Succesful creation of a train set and a test set")

    finally:
        # Closing the files
        original_dataset.close()
        train_set_h5.close()
        test_set_h5.close()


def _compute_distance(pt1, pt2):
    """
    Returns the distance between two points from their coordinates as numpy arrays of shape (1, 3)
    :param pt1: First point coordinates
    :param pt2: Second point coordinates
    :return: The distance between the two points
    """

    return np.sqrt(np.sum(np.square(np.diff(np.array([pt1, pt2]), axis=0))))


def _compute_distances(pt, ref_pt1, ref_pt2):
    """
    Returns an array (1, 2) containing the distance of one point to two reference points
    :param pt: point whose distances must be computed
    :param ref_pt1: first reference point
    :param ref_pt2: second reference point
    :return: the distances two the two reference points
    """
    return np.array([_compute_distance(pt, ref_pt1), _compute_distance(pt, ref_pt2)])


def _exists_bond(pt1, pt2, min_bond_length, max_bond_length):
    """
    Whether or not there exists a bond between two atoms
    :param pt1: First atom
    :param pt2: Second atom
    :param min_bond_length : minimal acceptable bond length
    :param max_bond_length : maximal acceptable bond length
    :return: True if there exists a bond, False otherwise
    """

    dist = _compute_distance(pt1, pt2)
    return min_bond_length < dist < max_bond_length


def _get_pos_class(pt, pos_a1, pos_a2):
    """
    Returns the positional class of the given point relatively to the positions of the two atoms of the couple
    :param pt: point whose positional class must be determined
    :param pos_a1: first atom of the couple
    :param pos_a2: second atom of the couple
    :return: positional class in one-hot-encoding style
    """

    # Output array initialization
    classes_pos = np.zeros(shape=(3,))

    # Computing vector A1_A2
    vect = np.diff([pos_a1, pos_a2], axis=0)

    # Declaration of the matrix containing the positions of points l, c, r
    gcd_pos = np.empty(shape=(3, 3))

    # Computing c point position
    gcd_pos[1] = np.divide([np.sum([pos_a1, pos_a2], axis=0)], 2)

    # Computing l and d positions
    gcd_pos[0] = gcd_pos[1] - vect
    gcd_pos[2] = gcd_pos[1] + vect

    # Computing matrix of repeated positions of the point
    pos = np.tile(pt, 3).reshape(3, 3)

    # Computing distances to points l, c and r
    dists = np.sqrt(np.sum(np.square(np.diff([gcd_pos, pos], axis=0)[0]), axis=1))

    # Returning the positional class in one-hot-encoding style
    classes_pos[np.argmin(dists)] = 1
    return classes_pos


def _get_anum_one_hot(z, max_anum):
    """
    Computes the atomic number in one-hot-encoding style
    :param z: numerical atomic number
    :param max_anum: max accepted atomic number
    :return: atomic number in one-hot-encoding style
    """

    if not 0 < z <= max_anum:
        raise RuntimeError("Atomic number must be between 1 and " + str(max_anum))

    one_hot = np.zeros(shape=(max_anum,))
    one_hot[int(z) - 1] = 1
    return one_hot


def _compute_input_width(max_anum, one_hot_anum, distances, pos_class, amasses):
    """
    Computes the model input width depending on the info we're putting in
    :param max_anum: max atomic number acceptable in the molecule
    :param one_hot_anum: whether or not we represent the atomic number of the atoms in one-hot-encoding style
    :param distances: whether or not the distances of each atom to both atom of the pair are put in the input
    :param pos_class: whether or not the positional class of each atom relatively to the pair is put in the input
    :param amasses: whether or not the atomic mass of each atom id put in the input
    :return: input width
    """
    return 0 + (one_hot_anum and max_anum) + (distances and 2) + (pos_class and 3) + (amasses and 1)


def _distance_constraint(at, at_ref1, at_ref2, cut_off_distance):
    return cut_off_distance is None or \
                                    _compute_distance(at, at_ref1) < cut_off_distance or \
                                    _compute_distance(at, at_ref2) < cut_off_distance


def _prepare_mol_data(coords_mol, anums_mol, amasses_mol, anum_1, anum_2, pubchem_id, min_bond_size, max_bond_size,
                      max_anum, max_mol_size, cut_off_distance=None, pos_class=True, one_hot_anum=True, amasses=True,
                      distances=True, distances_fun=None):
    """
    Preparing inputs and targets for the models from one molecule. Taking the data describing the molecule (coordinates,
    atomic numbers and atomic masses) and the atomic numbers of the couples of atoms whose distance of bonds must be
    predicted. Then we iterate over all the couples of atoms of specified atomic numbers in the molecule, and for those
    which share a bond, for each atom except for the two of the couple, we create an example for a model containing the
    specified information.

    :param coords_mol: Coordinates of the atoms of the molecule
    :param anums_mol: Atomic numbers of the atoms of the molecule
    :param amasses_mol: Atomic masses of the atoms of the molecule
    :param anum_1: Atomic number of the first atom of the couple we're interested in the bonds lengths
    :param anum_2: Atomic number of the second atom of the couple we're interested int the bonds lengths
    :param pubchem_id: Pubchem id of the molecule
    :param min_bond_size: Minimum length of bond between two atoms of a couple
    :param max_bond_size: Maximum length of bond between two atoms of a couple
    :param max_anum: max atomic number contained in the molecules we're extracting data from
    :param max_mol_size: max number of atoms that can be found in the molecule
    :param cut_off_distance: Radius of the sphere of center the center of the two atoms of a couple and inside which we
                             record information about the other atoms. If None, all the atoms of the molecule are
                             considered.
    :param pos_class: Whether or not the info about positional classes of the atoms is included in the model input
    :param one_hot_anum: Whether or not the info about the atomic number of the atoms is included in the model input
    :param amasses: Whether or not the atomic masses are included in the model input
    :param distances: Whether or not the distances to both of the atoms of the couple are included in the model input
    :return: inputs, targets and associated pubchem id for each input
    """

    mol_size = len(coords_mol)

    inputs_RN = []
    pubchem_ids = []
    targets_RN = []

    # Computing the width of each input depending on the included info
    input_rn_width = _compute_input_width(max_anum, one_hot_anum, distances, pos_class, amasses)

    # Iterating over all the atoms of the molecule
    for i in range(mol_size):

        # Selecting the first atom of the couple
        if anums_mol[i] == anum_1:

            # Iterating over all the following atoms
            for j in range(i + 1, mol_size):

                # Selecting the first atom of the couple if there exists a bond with the first one
                if anums_mol[j] == anum_2 and _exists_bond(coords_mol[i], coords_mol[j], min_bond_size, max_bond_size):

                    # Computing the distance between the two atoms of the couple
                    dist_couple = _compute_distance(coords_mol[i], coords_mol[j])

                    # Initialization of the RN input
                    input_rn = np.zeros(shape=(max_mol_size - 2, input_rn_width))

                    input_rn_idx = 0

                    # Iterating over all the atoms of the molecule (except for the couple that shares a bond)
                    for k in range(mol_size):
                        if k != i and k != j:

                            # If cut_off_distance is activated, checking if the
                            if cut_off_distance is None or \
                                    _compute_distance(coords_mol[k], coords_mol[j]) < cut_off_distance or \
                                    _compute_distance(coords_mol[k], coords_mol[i]) < cut_off_distance:

                                last_input_id = 0

                                # If included, recording the atomic number of the current atom
                                if one_hot_anum:
                                    input_rn[input_rn_idx][last_input_id:max_anum] = _get_anum_one_hot(anums_mol[k],
                                                                                                       max_anum)
                                    last_input_id += max_anum

                                # If included, recording the atomic mass of the current atom
                                if amasses:
                                    input_rn[input_rn_idx][last_input_id] = amasses_mol[k]
                                    last_input_id += 1

                                # If included, recording the distances to both of the atoms of the couple
                                if distances:
                                    input_rn[input_rn_idx][last_input_id:last_input_id + 2] = distances_fun(
                                        _compute_distances(
                                            coords_mol[k],
                                            coords_mol[i],
                                            coords_mol[j]))
                                    last_input_id += 2

                                # If included, recording the positional class of the atom
                                if pos_class:
                                    input_rn[input_rn_idx][last_input_id:last_input_id + 3] = _get_pos_class(
                                        coords_mol[k],
                                        coords_mol[i],
                                        coords_mol[j])
                                    last_input_id += 3

                                input_rn_idx += 1

                    # Flattening the input of the model
                    input_rn = input_rn.reshape(-1, )

                    # Recording the input for the current couple
                    inputs_RN.append(input_rn)

                    # Recording the target for the current couple
                    targets_RN.append(dist_couple * 1000)

                    # Recording the associated pubchem id to the current molecule
                    pubchem_ids.append(pubchem_id)

    return np.array(inputs_RN), np.array(targets_RN), np.array(pubchem_ids)


def identity_distances_fun(distances):
    """ Applies identity function to distances """
    return distances


def inv_distances_fun(distances):
    """ Applies inverse function to distances """
    return np.reciprocal(distances)


def squareinv_distances_fun(distances):
    """ Applies square inverse function to distances """
    return np.reciprocal(np.square(distances))


def generate_data(original_dataset_loc, prepared_input_loc, labels_loc, anum1, anum2, nb_mol, batch_size, max_anum,
                  min_bond_size, max_bond_size, min_mol_size, max_mol_size, cut_off_distance=None,
                  one_hot_anums=True, distances=True, pos_class=True, amasses=True, distances_fun_str=""):
    """
    Generates the prepared inputs and the targets for the specified dataset and the specified atoms couple from the
    nb_mol first molecules of the dataset

    :param original_dataset_loc: location of the original dataset
    :param prepared_input_loc: location of the prepared input dataset
    :param labels_loc: locations of the targets dataset
    :param anum1: atomic number of the first atom of the couple we want to predict the bonds lengths
    :param anum2: atomic number of the second atom of the couple we want to predict the bonds lengths
    :param nb_mol: number of molecules we extract we prepare data from in the original dataset
    :param batch_size: size of the batch
    :param max_anum: max atomic number contained in the molecules we're extracting data from
    :param one_hot_anums: whether or not the atomic numbers of the atoms are included in the input data (one-hot-encoding
                         style)
    :param distances: whether or not the distances of the atoms to the center of the couple's bond are included in the
                      input data
    :param pos_class: whether or not the positional classes of the atoms are included in the input data
    :param amasses: whether or not the masses of the atoms are included in the input data
    :param min_bond_size: minimal size of bond we consider that two atoms of a found couple share a bond
    :param max_bond_size: maximal size of bond we consider that two atoms of a found couple share a bond
    :param max_mol_size : maximal size of the molecules we extract data from
    :param cut_off_distance: Radius of the sphere of center the center of the two atoms of a couple and inside which
                             we record information about the other atoms. If None, all the atoms of the molecule are
                             considered.
    :return:
    """

    # Starting timer
    start_time = time.time()

    # Loading the input data
    original_dataset_h5 = h5py.File(original_dataset_loc, "r")

    # Creating input and labels files
    input_rn_dataset_h5 = h5py.File(prepared_input_loc, 'w')
    labels_dataset_h5 = h5py.File(labels_loc, 'w')

    # Selecting distances function
    if distances_fun_str == "identity":
        distances_fun = identity_distances_fun
    elif distances_fun_str == "inv":
        distances_fun = inv_distances_fun
    elif distances_fun_str == "squareinv":
        distances_fun = squareinv_distances_fun

    try:

        input_rn_width = _compute_input_width(max_anum, one_hot_anums, distances, pos_class, amasses)

        input_dataset = input_rn_dataset_h5.create_dataset(inputs_key, shape=(0, input_rn_width * (max_mol_size - 2)),
                                                           maxshape=(None, None),
                                                           dtype=np.float32, compression="gzip",
                                                           chunks=True)

        ids_dataset = input_rn_dataset_h5.create_dataset(pubchem_id_key, maxshape=(None, 1), shape=(0, 1),
                                                         dtype=np.int32, compression="gzip",
                                                         chunks=True)

        target_dataset = labels_dataset_h5.create_dataset(targets_key, maxshape=(None, 1), shape=(0, 1),
                                                          dtype=np.float32, compression="gzip",
                                                          chunks=True)

        # Initialization of indexes
        x_min_batch = 0
        x_max_batch = batch_size
        nb_total_mol = len(np.array(original_dataset_h5[anums_key][:nb_mol]))
        datasets_curr_idx = 0

        while x_min_batch <= nb_total_mol:

            print("New batch (molecules from " + str(x_min_batch) + " to " + str(min(x_max_batch, nb_total_mol)) +
                  ") --- " + str(float(min(x_min_batch, nb_total_mol) / nb_total_mol * 100)) + " %")

            # Loading data of current batch in memory
            input_coords = np.array(original_dataset_h5[coords_key][:nb_mol][x_min_batch:x_max_batch])
            input_nums = np.array(original_dataset_h5[anums_key][:nb_mol][x_min_batch:x_max_batch])
            input_masses = np.array(original_dataset_h5[amasses_key][:nb_mol][x_min_batch:x_max_batch])
            input_ids = np.array(original_dataset_h5[pubchem_id_key][:nb_mol][x_min_batch:x_max_batch])

            # Creating arrays for current batch
            inputs_batch = []
            ids_batch = []
            outputs_batch = []

            # Iterating over all the molecules of the current batch
            for batch_idx in range(len(input_nums)):

                # Ignoring molecules containing atoms of number above max_anum or containg too much atoms
                if max(input_nums[batch_idx]) <= max_anum and \
                        min_mol_size <= len(input_nums[batch_idx]) <= max_mol_size:

                    # Computing inputs and targets for the current molecule (there can be multiple input and targets
                    # if there exists several couples of the considered atoms in the molecule)
                    curr_inputs_np, curr_outputs_np, curr_ids_np = _prepare_mol_data(
                        input_coords[batch_idx].reshape(-1, 3),
                        input_nums[batch_idx],
                        input_masses[batch_idx], anum1, anum2,
                        input_ids[batch_idx], min_bond_size, max_bond_size,
                        max_anum, max_mol_size, cut_off_distance, pos_class, one_hot_anums,
                        amasses, distances, distances_fun)

                    # Adding inputs and targets to the arrays of the current batch
                    if len(curr_inputs_np) > 0:
                        inputs_batch.extend(curr_inputs_np)
                        outputs_batch.extend(curr_outputs_np)
                        ids_batch.extend(curr_ids_np)

            # End of the batch #

            # Resizing datasets
            input_dataset.resize((input_dataset.shape[0] + len(inputs_batch), input_rn_width * (max_mol_size - 2)))
            target_dataset.resize((target_dataset.shape[0] + len(inputs_batch), 1))
            ids_dataset.resize((ids_dataset.shape[0] + len(inputs_batch), 1))

            # Writing data to datasets
            input_dataset[datasets_curr_idx:datasets_curr_idx + len(inputs_batch)] = np.array(inputs_batch)
            target_dataset[datasets_curr_idx:datasets_curr_idx + len(inputs_batch)] = \
                                                                        np.array(outputs_batch).reshape(-1, 1)
            ids_dataset[datasets_curr_idx:datasets_curr_idx + len(inputs_batch)] = np.array(ids_batch).reshape(-1, 1)

            # Updating indexes
            datasets_curr_idx += len(inputs_batch)
            x_min_batch += batch_size
            x_max_batch += batch_size

        input_rn_dataset_h5.flush()
        labels_dataset_h5.flush()

        print(str(len(input_dataset)) + " created examples")
        print("--- %s seconds ---" % (time.time() - start_time))

    finally:
        input_rn_dataset_h5.close()
        labels_dataset_h5.close()
        original_dataset_h5.close()


def generate_data_wished_size(original_dataset_loc, prepared_input_loc, labels_loc, anum1, anum2, wished_size,
                              batch_size, max_anum, min_bond_size, max_bond_size, min_mol_size, max_mol_size,
                              cut_off_distance=None, one_hot_anums=True, distances=True, pos_class=True, amasses=True,
                              distances_fun_str = ""):
    """
    Generates a dataset of size that approximates a given wished size
    """

    mini_set_size = 500

    # Generation of a dataset on the mini_set_size first examples
    generate_data(original_dataset_loc, prepared_input_loc, labels_loc, anum1, anum2, mini_set_size, batch_size,
                  max_anum, min_bond_size, max_bond_size, min_mol_size, max_mol_size,
                  cut_off_distance=cut_off_distance, one_hot_anums=one_hot_anums, distances=distances,
                  pos_class=pos_class, amasses=amasses, distances_fun_str=distances_fun_str)

    # Computing the number of generated examples from mini_set_size molecules
    mini_set_prepared_input_h5 = h5py.File(prepared_input_loc, "r")
    mini_examples_nb = len(np.array(mini_set_prepared_input_h5[inputs_key]))

    mini_set_prepared_input_h5.close()

    # Computing the number of molecules that must be explored to obtain approximately wished_size examples
    nb_mol = math.floor((mini_set_size * wished_size) / mini_examples_nb)

    # Generation of the dataset
    generate_data(original_dataset_loc, prepared_input_loc, labels_loc, anum1, anum2, nb_mol, batch_size,
                  max_anum, min_bond_size, max_bond_size, min_mol_size, max_mol_size, cut_off_distance=cut_off_distance,
                  one_hot_anums=one_hot_anums, distances=distances, pos_class=pos_class, amasses=amasses,
                  distances_fun_str=distances_fun_str)
