import matplotlib.pyplot as plt
import matplotlib as mpl
from math import floor
import h5_keys
import numpy as np
from matplotlib import gridspec
from h5_keys import *
import h5py
import os

mpl.rcParams['agg.path.chunksize'] = 10000


def plot_distrib_rmses_val(rmses, model_name, figures_loc, display_plot):
    """ Display errors distribution """

    fig = plt.figure()

    plt.axis("off")

    # Linear Y scale
    ax1 = fig.add_subplot(121)
    ax1.set_title(model_name + " model\n Errors distribution")
    ax1.set_xlabel("Absolute error (pm)")
    ax1.set_ylabel("Test set occurrences (linear scale)")
    ax1.hist(rmses, floor(max(rmses)-min(rmses)))

    # Logarithmic Y scale
    ax2 = fig.add_subplot(122)
    ax2.set_yscale("log")
    ax2.set_title(model_name + " model\n Errors distribution")
    ax2.set_xlabel("Absolute error (pm)")
    ax2.set_ylabel("Test set occurrences (logarithmic scale)")

    ax2.hist(rmses, floor(max(rmses) - min(rmses)))

    plt.gcf().subplots_adjust(wspace=0.3)

    plt.savefig(figures_loc + model_name + "_distrib_rmse_val.png", dpi=250)

    if display_plot:
        plt.show()


def _colorbar_bonds_lengths_representation(ax, targets, bonds_lengths_loc):
    """
    Plots the colorbar representing the bonds lengths representation
    :param ax:
    :param targets:
    :param bonds_lengths_loc:
    :return:
    """
    bonds_lengths_h5 = h5py.File(bonds_lengths_loc, "r")
    bonds_lengths = np.array(bonds_lengths_h5[distances_key])
    hist_bonds = np.histogram(bonds_lengths*100, np.arange(min(targets), max(targets), 0.001))[0]

    cmap = mpl.cm.bwr_r

    norm = mpl.colors.Normalize()

    ax.set_xticklabels([])

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, values=hist_bonds,
                                    norm=norm,
                                    orientation='horizontal')

    cb1.set_label('Bond lengths representation')

def _print_typical_bonds_lengths(ax, anum_1, anum_2):

    if anum_1 == 6 and anum_2 == 6:

        x_axe_coord = ax.transAxes.inverted().transform(ax.transData.transform((1200, 0)))[0]
        ax.annotate('triple bounds', xy=(x_axe_coord, 0), xytext=(x_axe_coord - 0.1, 0.1), xycoords=ax.transAxes,
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    )

        x_axe_coord = ax.transAxes.inverted().transform(ax.transData.transform((1340, 0)))[0]
        ax.annotate('double bounds', xy=(x_axe_coord, 0), xytext=(x_axe_coord, 0.1), xycoords=ax.transAxes,
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    )

        x_axe_coord = ax.transAxes.inverted().transform(ax.transData.transform((1540, 0)))[0]
        ax.annotate('single bonds', xy=(x_axe_coord, 0), xytext=(x_axe_coord, 0.1), xycoords=ax.transAxes,
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    )

        x_axe_coord = ax.transAxes.inverted().transform(ax.transData.transform((1400, 0)))[0]
        ax.annotate('aromatic bonds', xy=(x_axe_coord, 0), xytext=(x_axe_coord, 0.1), xycoords=ax.transAxes,
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    )

    elif anum_1 == 6 and anum_2 == 1:

        x_axe_coord = ax.transAxes.inverted().transform(ax.transData.transform((1090, 0)))[0]
        ax.annotate('simple', xy=(x_axe_coord, 0), xytext=(x_axe_coord, 0.1), xycoords=ax.transAxes,
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    )

    elif anum_1 == 8 and anum_2 == 1:

        x_axe_coord = ax.transAxes.inverted().transform(ax.transData.transform((980, 0)))[0]
        ax.annotate('simple', xy=(x_axe_coord, 0), xytext=(x_axe_coord, 0.1), xycoords=ax.transAxes,
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    )

def plot_rmse_distrib_dist(rmses, targets, model_name, figures_loc, bonds_lengths_loc, display_plot, anum_1, anum_2):

    gs = gridspec.GridSpec(2, 1, height_ratios=[13, 1])

    ax = plt.subplot(gs[0])

    ax.set_title(model_name + " model " + "\nError distribution depending on target distances")
    ax.set_xlabel("Target distance (pm)")
    ax.set_ylabel("Absolute error (pm)")
    ax.plot(targets, rmses, ",", label="Absolute error (pm)", alpha=0.8)
    ax.set_xlim(xmin=min(targets), xmax=max(targets))

    ax2 = plt.subplot(gs[1])

    _colorbar_bonds_lengths_representation(ax2, targets, bonds_lengths_loc)

    plt.tight_layout()

    plt.savefig(figures_loc + model_name + "_distrib_rmse_dist.png", dpi=250)

    if display_plot:
        plt.show()


def fun_id(x):
    return x


def plot_targets_pred(targets, preds, anum_1, anum_2, model_name, figures_loc, bonds_lengths_loc, display_plot):
    gs = gridspec.GridSpec(2, 1, height_ratios=[13, 1])

    # Predictions depending on target distances plot
    ax = plt.subplot(gs[0])
    ax.set_title(model_name + " model\n Predictions depending on target distances")
    ax.set_xlabel("Target distance (pm)")
    ax.set_ylabel("Predicted distance (pm)")
    ax.plot(targets, preds, ",")
    ax.set_xlim(xmin=min(targets), xmax=max(targets))

    # Perfect model plot
    x = np.linspace(min(targets), max(targets))
    ax.plot(x, fun_id(x), color='darkgreen', label="Theoretical perfect model")
    ax.legend(loc='lower center', shadow=False)

    # Distances representation plot
    ax2 = plt.subplot(gs[1])
    _colorbar_bonds_lengths_representation(ax2, targets, bonds_lengths_loc)

    plt.tight_layout()

    plt.savefig(figures_loc + model_name + "_preds_targets.png", dpi=250)

    if display_plot:
        plt.show()


def print_stats_errors(errors):
    print("Dataset size : "+str(len(errors)))
    print("Mean error : " + str(np.mean(errors)))
    print("Median error : " + str(np.median(errors)))
    print("Standard deviation : " + str(np.std(errors)))
    print("Min error : " + str(min(errors)))
    print("Max error : " + str(max(errors)))


def plot_model_results(errors, predictions, targets, model_name, anum_1, anum_2, bonds_lengths_loc, plots_dir,
                       plot_error_distrib, plot_targets_error_distrib, plot_targets_predictions, display_plots):

    print("Plotting "+model_name)

    print_stats_errors(errors)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    if plot_error_distrib:
        plot_distrib_rmses_val(errors, model_name, plots_dir, display_plots)

    if plot_targets_error_distrib:
        plot_rmse_distrib_dist(errors, targets, model_name, plots_dir, bonds_lengths_loc, display_plots, anum_1,
                               anum_2)

    if plot_targets_predictions:
        plot_targets_pred(targets, predictions, anum_1, anum_2, model_name, plots_dir, bonds_lengths_loc, display_plots)


def biggest_errors_CIDs(errors, test_prepared_input_loc, n):
    """
    Prints the CIDs the molecules causing the n biggest errors
    :param errors:
    :param predictions:
    :param targets:
    :return:
    """

    cids = np.array(h5py.File(test_prepared_input_loc, 'r')[h5_keys.pubchem_id_key])