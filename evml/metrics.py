import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
import os


def compute_results(
    df,
    output_cols,
    mu,
    aleatoric,
    epistemic,
    legend_cols=["Friction_velocity", "Sensible_heat", "Latent_heat"],
    fn=None,
):
    mu_cols = [f"{x}_pred" for x in output_cols]
    err_cols = [f"{x}_err" for x in output_cols]
    e_cols = [f"{x}_e" for x in output_cols]
    a_cols = [f"{x}_a" for x in output_cols]
    # Add the predictions to the dataframe and compute absolute error
    df[mu_cols] = mu
    df[a_cols] = aleatoric
    df[e_cols] = epistemic
    df[err_cols] = np.abs(mu - df[output_cols])

    # Make 2D histogram of the predicted aleatoric and epistemic uncertainties
    try:
        plot_uncertainties(
            aleatoric, epistemic, output_cols, legend_cols=legend_cols, save_location=fn
        )
    except:
        pass
    # Compute attributes figure
    try:
        regression_attributes(df, output_cols, legend_cols, save_location=fn)
    except:
        pass
    # Compute calibration curve and MAE versus sorted epistemic uncertainty
    try:
        calibration(df, a_cols, e_cols, err_cols, legend_cols, save_location=fn)
    except Exception as E:
        print(E)
        pass
    # Compute calibration curve and MAE versus sorted aleatoric uncertainty
    # try:
    #     calibration(df, a_cols, err_cols, legend_cols, "Aleatoric", save_location=fn)
    # except:
    #     pass
    # spread-skill
    try:
        plot_skill_score(
            df[output_cols].values,
            mu,
            aleatoric,
            epistemic,
            output_cols,
            legend_cols=legend_cols,
            num_bins=20,
            save_location=fn
        )
        #spread_skill(df, output_cols, legend_cols, save_location=fn)
    except:
        pass
    # discard fraction
    try:
        discard_fraction(df, output_cols, legend_cols, save_location=fn)
    except:
        pass
    # Compute PIT histogram
    try:
        pit_histogram(
            df, mu, output_cols, num_bins=10, legend_cols=legend_cols, save_location=fn
        )
    except:
        pass


def compute_coverage(df, col="var", quan="error"):
    df = df.copy()
    df = df.sort_values(col, ascending=True)
    df["dummy"] = 1
    df[f"cu_{quan}"] = df[quan].cumsum() / df["dummy"].cumsum()
    df[f"cu_{col}"] = df[col].cumsum() / df["dummy"].cumsum()
    df[f"{col}_cov"] = 1 - df["dummy"].cumsum() / len(df)
    return df


def calibration_curve(df, col="var", quan="error", bins=10):
    obs = df.sort_values(quan, ascending=True).copy()
    obs[f"{quan}_cov"] = 1 - obs["dummy"].cumsum() / len(obs)
    h, b1, b2 = np.histogram2d(obs[f"{col}_cov"], obs[f"{quan}_cov"], bins=bins)
    cov_var = np.arange(0.025, 1.025, 1.0 / float(bins))
    cov_mae = [np.average(cov_var, weights=hi) for hi in h]
    cov_mae_std = [np.average((cov_mae - cov_var) ** 2, weights=hi) for hi in h]
    cov_var_std = [np.average((cov_mae - cov_var) ** 2, weights=hi) for hi in h.T]
    return cov_var, cov_mae, cov_mae_std, cov_var_std


def calibration(
    dataframe, a_cols, e_cols, mae_cols, legend_cols, bins=10, save_location=False
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5))
    colors = ["r", "g", "b"]
    lcolors = ["pink", "lightgreen", "lightblue"]

    for a_col, e_col, mae_col, col, lcol in zip(a_cols, e_cols, mae_cols, colors, lcolors):
        #  Coverage (sorted uncertainty) versus cumulative metric
        df = compute_coverage(dataframe, col=a_col, quan=mae_col)
        ax1.plot(df[f"{a_col}_cov"], df[f"cu_{mae_col}"], zorder=2, color=col)
        
        df = compute_coverage(dataframe, col=e_col, quan=mae_col)
        ax2.plot(df[f"{e_col}_cov"], df[f"cu_{mae_col}"], zorder=2, color=col)
        
        dataframe["tot_uncertainty"] = dataframe[f"{a_col}"] + dataframe[f"{e_col}"]
        df = compute_coverage(dataframe, col="tot_uncertainty", quan=mae_col)
        ax3.plot(df[f"tot_uncertainty_cov"], df[f"cu_{mae_col}"], zorder=2, color=col)

#         cov_var, cov_mae, cov_mae_std, cov_var_std = calibration_curve(
#             df, col=e_col, quan=mae_col, bins=bins
#         )

#         ax2.plot(cov_var, cov_mae, f"{col}-o")
#         ax2.errorbar(
#             cov_var,
#             cov_mae,
#             xerr=cov_var_std,
#             yerr=cov_mae_std,
#             capsize=0,
#             c=col,
#             elinewidth=3,
#             ecolor=lcol,
#         )

    ax1.set_xlabel("Confidence percentile (Aleatoric)")
    ax2.set_xlabel("Confidence percentile (Epistemic)")
    ax3.set_xlabel("Confidence percentile (Total)")
    ax1.set_ylabel("MAE")
    

    ax1.legend(legend_cols)
    ax2.legend(legend_cols)
    ax3.legend(legend_cols)
    #ax2.plot(cov_var, cov_var, "k--")
    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, f"{name}_coverage_calibration.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_uncertainties(
    ale,
    epi,
    output_cols,
    num_bins=20,
    legend_cols=None,
    fontsize=10,
    save_location=None,
):

    width = 5 if len(output_cols) == 1 else 10
    height = 3.5 if len(output_cols) == 1 else 3.5
    fig, axs = plt.subplots(1, len(output_cols), figsize=(width, height))
    if len(output_cols) == 1:
        axs = [axs]

    if legend_cols is None:
        legend_cols = output_cols

    # Loop over each element in output_cols and create a hexbin plot
    for i, col in enumerate(output_cols):
        # Calculate the mean prediction for the current column
        aleatoric = ale[:, i]
        epistemic = epi[:, i]

        # Create the 2D histogram plot
        my_range = [
            [np.percentile(epistemic, 0), np.percentile(epistemic, 98)],
            [np.percentile(aleatoric, 0), np.percentile(aleatoric, 98)],
        ]
        hb = axs[i].hist2d(
            epistemic, aleatoric, bins=num_bins, cmap="inferno", range=my_range
        )

        # Set the axis labels
        axs[i].set_title(legend_cols[i], fontsize=fontsize)
        axs[i].set_xlabel("Aleatoric", fontsize=fontsize)
        if i == 0:
            axs[i].set_ylabel("Epistemic", fontsize=fontsize)
        axs[i].tick_params(axis="both", which="major", labelsize=fontsize)

        # Move the colorbar below the x-axis label
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = fig.colorbar(hb[3], cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)

        # Set the tick labels to use scientific notation
        axs[i].ticklabel_format(style="sci", axis="both", scilimits=(-1, 1))

    # make it pretty
    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, "compare_uncertanties.pdf"),
            dpi=300,
            bbox_inches="tight",
        )


# def spread_skill(df, output_cols, legend_cols, nbins=20, save_location=None):
#     colors = ["r", "g", "b"]
#     uncertainty_cols = ["e", "a"]
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     lower_bounds = defaultdict(list)
#     upper_bounds = defaultdict(list)
#     for k, col in enumerate(output_cols):
#         for j, u in enumerate(uncertainty_cols):

#             upper = max(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
#             lower = min(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])

#             #             upper = 1.01 * max(df[f"{col}_{u}"])
#             #             lower = 0.99 * min(df[f"{col}_{u}"])

#             #             lower = (
#             #                 lower
#             #                 if np.isfinite(lower)
#             #                 else 0.99 * min(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
#             #             )
#             #             upper = (
#             #                 upper
#             #                 if np.isfinite(upper)
#             #                 else 1.01 * max(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
#             #             )

#             if upper > 0 and lower > 0 and (np.log10(upper) - np.log10(lower)) > 2:
#                 bins = np.logspace(np.log10(lower), np.log10(upper), nbins)
#             else:
#                 bins = np.linspace(lower, upper, nbins)

#             bin_range = np.digitize(df[f"{col}_{u}"].values, bins=bins, right=True)
#             bin_means = [
#                 df[f"{col}_{u}"][bin_range == i].mean() for i in range(1, len(bins))
#             ]

#             histogram = defaultdict(list)
#             for bin_no in range(1, max(list(set(bin_range)))):
#                 idx = np.where(bin_range == bin_no)
#                 residuals = df[f"{col}_err"].values[idx] ** 2
#                 mean = np.mean(residuals) ** (1 / 2)
#                 std = np.std(residuals) ** (1 / 2)
#                 histogram["bin"].append(bin_means[bin_no - 1])
#                 histogram["mean"].append(mean)
#                 histogram["std"].append(std)

#             axs[j].errorbar(
#                 histogram["bin"], histogram["mean"], yerr=histogram["std"], c=colors[k]
#             )
#             axs[j].legend(legend_cols)
#             lower_bounds[u].append(bin_means[0])
#             upper_bounds[u].append(bin_means[-2])

#     bins = np.linspace(min(lower_bounds["e"]), max(upper_bounds["e"]), nbins)
#     axs[0].plot(bins, bins, color="k", ls="--")
#     bins = np.linspace(min(lower_bounds["a"]), max(upper_bounds["a"]), nbins)
#     axs[1].plot(bins, bins, color="k", ls="--")
#     axs[0].set_xlabel("Spread (Epistemic uncertainty)")
#     axs[1].set_xlabel("Spread (Aleatoric uncertainty)")
#     axs[0].set_ylabel("Skill score (RMSE)")

#     if len(output_cols) > 1:
#         axs[0].set_xscale("log")
#         axs[0].set_yscale("log")
#         axs[1].set_xscale("log")
#         axs[1].set_yscale("log")

#     plt.tight_layout()

#     if save_location:
#         plt.savefig(
#             os.path.join(save_location, "spread_skill.pdf"),
#             dpi=300,
#             bbox_inches="tight",
#         )


def compute_skill_score(y_true, y_pred, y_std, num_bins=10):
    """
    Computes the skill score with RMSE on the y-axis and binned spread on the x-axis.

    Parameters
    ----------
    y_true : array-like
        A 1D array of true values.
    y_pred : array-like
        A 1D array of predicted values.
    y_std : array-like
        A 1D array of standard deviations of predicted values.
    num_bins : int, optional
        The number of bins to use for binning the spread.

    Returns
    -------
    ss : array-like
        A 2D array of skill scores.
    bins : array-like
        A 1D array of bin edges for the spread.
    """

    # Bin the spread
    spread_min, spread_max = np.percentile(y_std, [5, 95])
    if spread_max - spread_min > 20:
        bins = np.geomspace(spread_min, spread_max, num_bins + 1)
    else:
        bins = np.linspace(spread_min, spread_max, num_bins + 1)
    digitized = np.digitize(y_std, bins)

    # Compute the mean RMSE for each bin
    ss = np.zeros((num_bins,))
    count = np.zeros((num_bins,))
    for i in range(num_bins):
        idx = np.where(digitized == i + 1)[0]
        if len(idx) > 0:
            ss[i] = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
            count[i] = len(idx)
    return ss, count, bins


def plot_skill_score(
    y_true, y_pred, y_ale, y_epi, output_cols, num_bins=50, legend_cols=None, save_location=False
):
    """
    Plots the skill score with RMSE on the y-axis and binned spread on the x-axis.

    Parameters
    ----------
    y_true : array-like
        A 1D array of true values.
    y_pred : array-like
        A 1D array of predicted values.
    y_std : array-like
        A 1D array of standard deviations of predicted values.
    num_bins : int, optional
        The number of bins to use for binning the spread.
    """
    num_outputs = len(output_cols)
    if num_outputs == 1:
        y_true = np.expand_dims(y_true, 1)
        y_pred = np.expand_dims(y_pred, 1)
        y_ale = np.expand_dims(y_ale, 1)
        y_epi = np.expand_dims(y_epi, 1)

    width = 10  # 5 if num_outputs == 1 else 10
    height = 3.5 if num_outputs == 1 else 7
    fig, axs = plt.subplots(num_outputs, 3, figsize=(width, height))
    if num_outputs == 1:
        axs = [axs]

    if legend_cols is None:
        legend_cols = output_cols

    unc_lab = ["Aleatoric", "Epistemic", "Total"]

    for j in range(num_outputs):

        y_tot = np.sqrt(y_ale**2 + y_epi**2)
        
        for i, std in enumerate([y_ale, y_epi, y_tot]):

            # Compute the skill score
            ss, counts, bins = compute_skill_score(
                y_true[:, j], y_pred[:, j], std[:, j], num_bins
            )

            # Compute bin centers
            x_centers = (bins[:-1] + bins[1:]) / 2
            y_centers = ss

            # Calculate range based on percentile of counts
            my_range = [
                [
                    min(np.percentile(x_centers, 5), np.percentile(y_centers, 5)),
                    np.percentile(x_centers, 95),
                ],
                [
                    min(np.percentile(x_centers, 5), np.percentile(y_centers, 5)),
                    np.percentile(y_centers, 95),
                ],
            ]

            _ = axs[j][i].hist2d(
                x_centers,
                y_centers,
                weights=counts / sum(counts),
                bins=num_bins,
                cmap="inferno",
                range=my_range,
            )

            # Add 1-1 line
            minx = min(min(x_centers), min(y_centers))
            maxx = max(max(x_centers), max(y_centers))
            ranger = np.linspace(minx, maxx, 10)
            axs[j][i].plot(ranger, ranger, c="b", ls="--", lw=3, zorder=10)
            axs[j][i].set_xlim([my_range[0][0], my_range[0][1]])
            axs[j][i].set_ylim([my_range[1][0], my_range[1][1]])

            if i == 0:
                axs[j][i].set_ylabel(f"{legend_cols[j]}\nSkill (RMSE)")
            if j == num_outputs - 1:
                axs[j][i].set_xlabel(f"Spread ({unc_lab[i]})")
            axs[j][i].ticklabel_format(style="sci", axis="both", scilimits=(-1, 1))

    plt.tight_layout()
    if save_location:
        plt.savefig(
            os.path.join(save_location, "spread_skill.pdf"),
            dpi=300,
            bbox_inches="tight",
        )



def discard_fraction(df, output_cols, legend_cols, save_location=False):
    width = 7 if len(output_cols) == 1 else 10
    height = 3.5
    fig, axs = plt.subplots(1, len(output_cols), figsize=(width, height))
    if len(output_cols) == 1:
        axs = [axs]
    for col in output_cols:
        df = compute_coverage(df, col=f"{col}_e", quan=f"{col}_err")
        df = compute_coverage(df, col=f"{col}_a", quan=f"{col}_err")
    for k, col in enumerate(output_cols):
        results = defaultdict(list)
        for percent in range(5, 105, 5):
            c = df[f"{col}_e_cov"] >= percent / 100.0
            results["rmse_e"].append(np.square(df[c][f"{col}_err"]).mean() ** (1 / 2))
            c = df[f"{col}_a_cov"] >= percent / 100.0
            results["rmse_a"].append(np.square(df[c][f"{col}_err"]).mean() ** (1 / 2))
            results["frac"].append(percent)

        axs[k].bar(results["frac"], results["rmse_e"], 2.5)
        axs[k].bar([x + 2.5 for x in results["frac"]], results["rmse_a"], 2.5)
        axs[k].set_xlabel("Fraction removed")
        axs[k].set_title(legend_cols[k])

        if k == 1:
            axs[k].legend(["Epistemic", "Aleatoric"])
    axs[0].set_ylabel("RMSE")
    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, "discard_fraction.pdf"),
            dpi=300,
            bbox_inches="tight",
        )


def regression_attributes(df, output_cols, legend_cols, nbins=11, save_location=False):
    width = 7 if len(output_cols) == 1 else 10
    height = 3.5
    fig, axs = plt.subplots(1, len(output_cols), figsize=(width, height))

    if len(output_cols) == 1:
        axs = [axs]

    for k, col in enumerate(output_cols):
        upper = 1.01 * max(df[f"{col}_pred"])
        lower = 0.99 * min(df[f"{col}_pred"])
        bins = np.linspace(lower, upper, nbins)
        bin_range = np.digitize(df[f"{col}_pred"].values, bins=bins)
        bin_means = [
            df[f"{col}_pred"][bin_range == i].mean() for i in range(1, len(bins))
        ]
        histogram = defaultdict(list)
        for bin_no in range(1, max(list(set(bin_range)))):
            idx = np.where(bin_range == bin_no)
            residuals = df[f"{col}"].values[idx]
            mean = np.mean(residuals)
            std = np.std(residuals)
            histogram["bin"].append(bin_means[bin_no - 1])
            histogram["mean"].append(mean)
            histogram["std"].append(std)
        axs[k].errorbar(histogram["bin"], histogram["mean"], yerr=histogram["std"])
        axs[k].plot(histogram["bin"], histogram["bin"], "k--")

        axs[k].set_title(f"{legend_cols[k]}")
        axs[k].set_ylabel("Conditional mean observation")
        axs[k].set_xlabel("Prediction")

    plt.tight_layout()
    if save_location:
        plt.savefig(
            os.path.join(save_location, "regression_attributes.pdf"),
            dpi=300,
            bbox_inches="tight",
        )


def compute_pit(true_dist, pred_dist):
    """
    Computes the probability integral transform (PIT) between a true distribution
    and a predicted distribution.

    Parameters
    ----------
    true_dist : array-like
        A 1D array of values representing the true distribution.
    pred_dist : array-like
        A 1D array of values representing the predicted distribution.

    Returns
    -------
    pit : array-like
        A 1D array of PIT values.
    """
    # Sort the true and predicted distributions
    true_sorted = np.sort(true_dist)
    pred_sorted = np.sort(pred_dist)

    # Compute the PIT values
    pit = np.zeros_like(pred_sorted)
    for i, x in enumerate(pred_sorted):
        j = np.searchsorted(true_sorted, x, side="right")
        pit[i] = j / len(true_sorted)

    return pit


def pit_histogram(
    df, pred_dist, output_cols, num_bins=50, legend_cols=None, save_location=None
):
    width = 5 if len(output_cols) == 1 else 10
    height = 3.5 if len(output_cols) == 1 else 3.5
    fig, axs = plt.subplots(1, len(output_cols), figsize=(width, height))
    if len(output_cols) == 1:
        axs = [axs]

    if legend_cols is None:
        legend_cols = output_cols
        
    true_dist = df[output_cols].values

    for k in range(true_dist.shape[-1]):
        td = true_dist[:, k]
        pd = pred_dist[:, k]

        # Compute the PIT values
        pit = compute_pit(td, pd)

        # Plot the histogram of PIT values
        bin_lims = np.percentile(pit, [0, 100])
        bins = np.linspace(bin_lims[0], bin_lims[1], num_bins + 1)
        hist, _ = np.histogram(pit, bins=bins, density=True)
        hist /= sum(hist)

        axs[k].bar(
            bins[:-1],
            hist,
            width=1.0 / (num_bins + 1),
            align="edge",
            color="tab:blue",
            edgecolor="lightblue",
        )
        axs[k].set_xlim(bin_lims[0], bin_lims[1])
        axs[k].set_ylim(0, 1.1 * np.max(hist))

        axs[k].set_xlabel("PIT", fontsize=12)
        axs[k].set_ylabel("Probability", fontsize=12)
        axs[k].set_title(legend_cols[k])

        # add baseline line ~ number of bins
        axs[k].plot(
            np.linspace(0, 1, num_bins),
            [1.0 / num_bins for x in range(num_bins)],
            color="k",
            ls="--",
        )

    plt.tight_layout()
    if save_location:
        plt.savefig(
            os.path.join(save_location, "pit_histogram.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
