import matplotlib.pyplot as plt
import matplotlib as mpl
from math import floor
import model_nn

mpl.rcParams['agg.path.chunksize'] = 10000


def plot_distrib_rmses_val(rmses, model_name, figures_loc):
    """ Display errors distribution according to a given padding """

    fig = plt.figure()

    print("rmses : ")
    print(rmses[:20])

    # Linear Y scale
    ax1 = fig.add_subplot(121)
    ax1.set_title(model_name + "model\n Errors distribution")
    ax1.set_xlabel("Absolute error (mÅ)")
    ax1.set_ylabel("Test set occurrences")
    ax1.hist(rmses, floor(max(rmses)-min(rmses)))

    # Logarithmic Y scale
    ax2 = fig.add_subplot(122)
    ax2.set_yscale("log")
    ax2.set_title(model_name + "model\n Errors distribution")
    ax2.set_xlabel("Absolute error (mÅ)")
    ax2.set_ylabel("Test set occurrences")
    ax2.hist(rmses, floor(max(rmses) - min(rmses)))

    plt.gcf().subplots_adjust(wspace=0.3)

    plt.savefig(figures_loc + model_name + "_distrib_rmse_val.png", dpi=250)
    plt.show()


def plot_nn_model_results(model_loc, model_name, anum_1, anum_2, bonds_lengths_loc,
                       test_prepared_input_loc, test_labels_loc, plots_dir, plot_error_distrib,
                       plot_targets_error_distrib, plot_targets_predictions, batch_size, last_layer_width, depth):

    errors, predictions, targets = model_nn.predict(model_loc, test_prepared_input_loc, test_labels_loc,
                                                    batch_size, last_layer_width, depth)

    if plot_error_distrib:
        plot_distrib_rmses_val(errors, model_name, plots_dir)
