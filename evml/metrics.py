import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os


def compute_results(df, output_cols, mu, aleatoric, epistemic, fn=None):
    mu_cols = [f"{x}_pred" for x in output_cols]
    err_cols = [f"{x}_err" for x in output_cols]
    e_cols = [f"{x}_e" for x in output_cols]
    a_cols = [f"{x}_a" for x in output_cols]
    # Add the predictions to the dataframe and compute absolute error
    df[mu_cols] = mu
    df[a_cols] = aleatoric
    df[e_cols] = epistemic
    df[err_cols] = np.abs(mu - df[output_cols])
    legend_cols = ["Friction_velocity", "Sensible_heat", "Latent_heat"]

    # Compute attributes figure
    try:
        regression_attributes(df, output_cols, save_location=fn)
    except:
        pass
    # Compute calibration curve and MAE versus sorted epistemic uncertainty
    try:
        calibration(df, e_cols, err_cols, legend_cols, "Epistemic", save_location=fn)
    except:
        pass
    # Compute calibration curve and MAE versus sorted aleatoric uncertainty
    try:
        calibration(df, a_cols, err_cols, legend_cols, "Aleatoric", save_location=fn)
    except:
        pass
    # spread-skill
    try:
        spread_skill(df, output_cols, legend_cols, save_location=fn)
    except:
        pass
    # discard fraction
    try:
        discard_fraction(df, output_cols, legend_cols, save_location=fn)
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
    dataframe, e_cols, mae_cols, legend_cols, name, bins=10, save_location=False
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colors = ["r", "g", "b"]
    lcolors = ["pink", "lightgreen", "lightblue"]

    for e_col, mae_col, col, lcol in zip(e_cols, mae_cols, colors, lcolors):
        #  Coverage (sorted uncertainty) versus cumulative metric
        df = compute_coverage(dataframe, col=e_col, quan=mae_col)
        ax1.plot(df[f"{e_col}_cov"], df[f"cu_{mae_col}"], zorder=2, color=col)

        cov_var, cov_mae, cov_mae_std, cov_var_std = calibration_curve(
            df, col=e_col, quan=mae_col, bins=bins
        )

        ax2.plot(cov_var, cov_mae, f"{col}-o")
        ax2.errorbar(
            cov_var,
            cov_mae,
            xerr=cov_var_std,
            yerr=cov_mae_std,
            capsize=0,
            c=col,
            elinewidth=3,
            ecolor=lcol,
        )

    ax1.set_xlabel("Confidence percentile")
    ax1.set_ylabel("MAE")

    ax2.set_xlabel(f"Estimated confidence percentile ({name})")
    ax2.set_ylabel("Observed confidence percentile (MAE)")

    ax1.legend(legend_cols)
    ax2.legend(legend_cols)
    ax2.plot(cov_var, cov_var, "k--")
    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, f"{name}_coverage_calibration.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def spread_skill(df, output_cols, legend_cols, nbins=20, save_location=None):
    colors = ["r", "g", "b"]
    uncertainty_cols = ["e", "a"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    lower_bounds = defaultdict(list)
    upper_bounds = defaultdict(list)
    for k, col in enumerate(output_cols):
        for j, u in enumerate(uncertainty_cols):
            
            upper = max(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
            lower = min(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
            
#             upper = 1.01 * max(df[f"{col}_{u}"])
#             lower = 0.99 * min(df[f"{col}_{u}"])
            
#             lower = (
#                 lower
#                 if np.isfinite(lower)
#                 else 0.99 * min(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
#             )
#             upper = (
#                 upper
#                 if np.isfinite(upper)
#                 else 1.01 * max(df[f"{col}_{u}"][~df[f"{col}_{u}"].isna()])
#             )
            
            if upper > 0 and lower > 0 and (np.log10(upper) - np.log10(lower)) > 2:
                bins = np.logspace(np.log10(lower), np.log10(upper), nbins)
            else:
                bins = np.linspace(lower, upper, nbins)
            
            bin_range = np.digitize(df[f"{col}_{u}"].values, bins=bins, right=True)
            bin_means = [
                df[f"{col}_{u}"][bin_range == i].mean() for i in range(1, len(bins))
            ]
        
            histogram = defaultdict(list)
            for bin_no in range(1, max(list(set(bin_range)))):
                idx = np.where(bin_range == bin_no)
                residuals = df[f"{col}_err"].values[idx] ** 2
                mean = np.mean(residuals) ** (1 / 2)
                std = np.std(residuals) ** (1 / 2)
                histogram["bin"].append(bin_means[bin_no - 1])
                histogram["mean"].append(mean)
                histogram["std"].append(std)

            axs[j].errorbar(
                histogram["bin"], histogram["mean"], yerr=histogram["std"], c=colors[k]
            )
            axs[j].legend(legend_cols)
            lower_bounds[u].append(bin_means[0])
            upper_bounds[u].append(bin_means[-2])
            
    bins = np.linspace(min(lower_bounds["e"]), max(upper_bounds["e"]), nbins)
    axs[0].plot(bins, bins, color="k", ls="--")
    bins = np.linspace(min(lower_bounds["a"]), max(upper_bounds["a"]), nbins)
    axs[1].plot(bins, bins, color="k", ls="--")
    axs[0].set_xlabel("Spread (Epistemic uncertainty)")
    axs[1].set_xlabel("Spread (Aleatoric uncertainty)")
    axs[0].set_ylabel("Skill score (RMSE)")

    # axs[0].set_xscale("log")
    # axs[0].set_yscale("log")
    # axs[1].set_xscale("log")
    # axs[1].set_yscale("log")

    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, "spread_skill.pdf"),
            dpi=300,
            bbox_inches="tight",
        )


def discard_fraction(df, output_cols, legend_cols, save_location=False):
    width = 7 if len(output_cols) == 1 else 10
    height = 5
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


def regression_attributes(df, output_cols, nbins=11, save_location=False):
    width = 7 if len(output_cols) == 1 else 10
    height = 5
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

        axs[k].set_title(f"{col}")
        axs[k].set_ylabel("Conditional mean observation")
        axs[k].set_xlabel("Prediction")

    plt.tight_layout()
    if save_location:
        plt.savefig(
            os.path.join(save_location, "regression_attributes.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
