import h5py
from h5_keys import *
import numpy as np


def _compute_distances(pt1, pt2):
    """
    Returns the distance between the two points represented by arrays of size (1, 3)
    :param pt1: first point coordinates
    :param pt2: second point coordinates
    :return: the distance between the two points
    """
    return np.sqrt(np.sum(np.square(np.diff(np.array([pt1, pt2]), axis=0))))


def _distances_mol(coords_mol, anums_mol, anum_1, anum_2):
    """
    Computes the distances between the pairs of the given atomic numbers in the molecule

    :param coords_mol: coordinates
    :param anums_mol: atomic numbers
    :param anum_1: first atomic number of the pair
    :param anum_2: second atomic number of the pair
    :return: distances between the atoms of the specified pair of atomic numbers
    """

    mol_size = len(coords_mol)

    dists = []

    # Iterating over all the atoms of the molecule
    for i in range(mol_size):

        # Selecting the first atom of the pair
        if anums_mol[i] == anum_1:

            # Iterating over all the following atoms
            for j in range(i + 1, mol_size):

                # Selecting the second atom of the pair
                if anums_mol[j] == anum_2:
                    # Computing distance between the two atoms of the pair
                    dists.append(_compute_distances(coords_mol[i], coords_mol[j]))

    return np.array(dists)


def record_bonds_lengths(dataset_loc, output_lengths_loc, anum_1, anum_2, batch_size, max_anum, min_mol_size,
                         max_mol_size):
    """
    Records all the distances between the pairs of specified atoms of a dataset in a h5 file

    :param dataset_loc: original dataset location
    :param output_lengths_loc: h5 file in which the distances are stored
    :param anum_1: first atomic number of the couple of atoms we're interested in the bonds lengths
    :param anum_2: second atomic number of the couple of atoms we're interested in the bonds lengths
    :param batch_size: max number of molecules that are simultaneously stored in memory
    :param max_anum: max atomic number that can be present in molecules we are extracting the bonds lengths from
    :param min_mol_size: min molecule size we're extracting the bonds lengths from
    :param max_mol_size: max molecule size we're extracting the bonds lengths from
    :return:
    """

    # Opening the input file
    original_dataset_h5 = h5py.File(dataset_loc, "r")

    # Creating the distances file
    distances_h5 = h5py.File(output_lengths_loc, 'w')

    # Computing the total number of molecules
    total_mol_nb = len(np.array(original_dataset_h5[anums_key]))

    try:

        # Creation of output dataset
        distances_dataset = distances_h5.create_dataset(distances_key, maxshape=(None,), shape=(0,),
                                                        dtype=np.float32, compression="gzip",
                                                        chunks=True)

        # Initialization of indexes
        x_min_batch = 0
        x_max_batch = batch_size
        dataset_curr_idx = 0

        # Initialization of computed distances counter
        nb_dist_tot = 0

        # Computing distances on each batch
        while x_min_batch < total_mol_nb:

            print("New batch (molecules from idx " + str(x_min_batch) + " to " + str(min(x_max_batch, total_mol_nb)) +
                  ") --- " + str(float(min(x_max_batch, total_mol_nb) / total_mol_nb * 100)) + " %")

            # Loading coordinates and atomic numbers of the molecules of the current batch
            input_coords = np.array(original_dataset_h5[coords_key][x_min_batch:x_max_batch])
            input_nums = np.array(original_dataset_h5[anums_key][x_min_batch:x_max_batch])

            # Creating python list of distances for the current batch
            distances_batch = []

            # Iterating over all the molecules of the batch
            for batch_idx in range(len(input_nums)):

                # Only caring about molecules that respect the given criterias
                if min_mol_size <= len(input_nums[batch_idx]) <= max_mol_size\
                                    and max(input_nums[batch_idx]) <= max_anum:

                    # Computing the distances of the molecule
                    distances_batch.extend(_distances_mol(input_coords[batch_idx].reshape(-1, 3),
                                                          input_nums[batch_idx], anum_1, anum_2))

            # Recording the data for the current batch
            distances_dataset.resize((distances_dataset.shape[0] + len(distances_batch),))  # resizing
            distances_dataset[dataset_curr_idx:dataset_curr_idx + len(distances_batch)] = np.array(distances_batch)

            # Updating indexes
            dataset_curr_idx += len(distances_batch)
            x_min_batch += batch_size
            x_max_batch += batch_size

            # Updating the total number of computed distances
            nb_dist_tot += len(distances_batch)

            print("... " + str(len(distances_batch)) + " computed distances")

        distances_h5.flush()

        print(str(nb_dist_tot) + " computed distances (total)")

    finally:
        original_dataset_h5.close()
        distances_h5.close()