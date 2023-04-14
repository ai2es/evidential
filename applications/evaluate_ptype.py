import logging
import yaml
import shutil
import os
import glob
import warnings
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from ptype.reliability import (
    compute_calibration,
    reliability_diagram,
    reliability_diagrams,
)
from ptype.plotting import (
    plot_confusion_matrix,
    coverage_figures,
)
from evml.metrics import compute_results_categorical
from evml.classifier_uq import uq_results

from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
from hagelslag.evaluation.MetricPlotter import roc_curve, performance_diagram

from collections import OrderedDict, defaultdict


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def locate_best_model(filepath, metric="val_ave_acc", direction="max"):
    filepath = glob.glob(os.path.join(filepath, "models", "training_log_*.csv"))
    func = min if direction == "min" else max
    scores = defaultdict(list)
    for filename in filepath:
        f = pd.read_csv(filename)
        best_ensemble = int(filename.split("_log_")[1].strip(".csv"))
        scores["best_ensemble"].append(best_ensemble)
        scores["metric"].append(func(f[metric]))

    best_c = scores["metric"].index(func(scores["metric"]))
    return scores["best_ensemble"][best_c]


def evaluate(conf, reevaluate=False):
    output_features = conf["ptypes"]
    n_splits = conf["n_splits"]
    save_loc = conf["save_loc"]
    labels = ["rain", "snow", "sleet", "frz-rain"]
    sym_colors = ["blue", "grey", "red", "purple"]
    symbols = ["s", "o", "v", "^"]

    if n_splits == 1:
        data = {
            name: pd.read_parquet(os.path.join(save_loc, "evaluate", f"{name}.parquet"))
            for name in ["train", "val", "test"]
        }
    else:
        best_split = locate_best_model(save_loc, conf["metric"], conf["direction"])
        data = {
            name: pd.read_parquet(
                os.path.join(save_loc, "evaluate", f"{name}_{best_split}.parquet")
            )
            for name in ["train", "val", "test"]
        }
        # Loop over the data splits
        for name in ["test"]:
            ensemble_p = np.zeros((n_splits, data[name].shape[0]))
            ensemble_std = np.zeros((n_splits, data[name].shape[0]))
            ensemble_entropy = np.zeros((n_splits, data[name].shape[0]))
            ensemble_mutual = np.zeros((n_splits, data[name].shape[0]))

            # Loop over ensemble of parametric models
            for split in range(n_splits):
                dfe = pd.read_parquet(
                    os.path.join(save_loc, "evaluate", f"{name}_{split}.parquet")
                )
                ensemble_p[split] = dfe["pred_conf"]
                ensemble_std[split] = dfe["epistemic"]
                ensemble_entropy[split] = dfe["entropy"]
                ensemble_mutual[split] = dfe["mutual_info"]

            # Compute averages, uncertainties
            data[name]["ave_conf"] = np.mean(ensemble_p, axis=0)
            data[name]["ave_entropy"] = np.mean(ensemble_entropy, axis=0)
            data[name]["ave_mutual_info"] = np.mean(ensemble_mutual, axis=0)
            data[name]["epistemic"] = np.var(ensemble_p, axis=0)
            data[name]["aleatoric"] = np.mean(ensemble_std, axis=0)

    # Compute categorical metrics
    metrics = defaultdict(list)
    for name in data.keys():
        outs = precision_recall_fscore_support(
            data[name]["true_label"].values,
            data[name]["pred_label"].values,
            average=None,
            labels=range(len(output_features)),
        )
        metrics["split"].append(name)
        for i, (p, r, f, s) in enumerate(zip(*list(outs))):
            class_name = output_features[i]
            metrics[f"{class_name}_precision"].append(p)
            metrics[f"{class_name}_recall"].append(r)
            metrics[f"{class_name}_f1"].append(f)
            metrics[f"{class_name}_support"].append(s)

    # Confusion matrix
    plot_confusion_matrix(
        data,
        labels,
        normalize=True,
        save_location=os.path.join(save_loc, "plots", "confusion_matrices.pdf"),
    )

    # Reliability
    metric_keys = [
        "avg_accuracy",
        "avg_confidence",
        "expected_calibration_error",
        "max_calibration_error",
    ]
    for name in data.keys():
        # Calibration stats
        results_calibration = compute_calibration(
            data[name]["true_label"].values,
            data[name]["pred_label"].values,
            data[name]["pred_conf"].values,
            num_bins=10,
        )
        for key in metric_keys:
            metrics[f"bulk_{key}"].append(results_calibration[key])
        # Bulk
        _ = reliability_diagram(
            data[name]["true_label"].values,
            data[name]["pred_label"].values,
            data[name]["pred_conf"].values,
            num_bins=10,
            dpi=300,
            return_fig=True,
        )
        fn = os.path.join(save_loc, "plots", f"bulk_reliability_{name}.pdf")
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        # Class by class
        results = OrderedDict()
        for label in range(len(output_features)):
            cond = data[name]["true_label"] == label
            results[output_features[label]] = {
                "true_labels": data[name][cond]["true_label"].values,
                "pred_labels": data[name][cond]["pred_label"].values,
                "confidences": data[name][cond]["pred_conf"].values,
            }
            results_calibration = compute_calibration(
                results[output_features[label]]["true_labels"],
                results[output_features[label]]["pred_labels"],
                results[output_features[label]]["confidences"],
                num_bins=10,
            )
            for key in metric_keys:
                metrics[f"{output_features[label]}_{key}"].append(
                    results_calibration[key]
                )

        _ = reliability_diagrams(
            results,
            num_bins=10,
            draw_bin_importance="alpha",
            num_cols=2,
            dpi=100,
            return_fig=True,
        )
        fn = os.path.join(save_loc, "plots", f"class_reliability_{name}.pdf")
        plt.savefig(fn, dpi=300, bbox_inches="tight")

    # Hagelslag
    for name in data.keys():
        rocs = []
        for i in range(len(output_features)):
            forecasts = data[name]["pred_conf"]
            obs = np.where(data[name]["true_label"] == i, 1, 0)
            roc = DistributedROC(
                thresholds=np.arange(0.0, 1.01, 0.01), obs_threshold=0.5
            )
            roc.update(forecasts, obs)
            rocs.append(roc)
            metrics[f"{output_features[i]}_auc"].append(roc.auc())
            metrics[f"{output_features[i]}_csi"].append(roc.max_csi())
        roc_curve(
            rocs,
            labels,
            sym_colors,
            symbols,
            os.path.join(save_loc, "plots", f"roc_curve_{name}.pdf"),
        )
        performance_diagram(
            rocs,
            labels,
            sym_colors,
            symbols,
            os.path.join(save_loc, "plots", f"performance_{name}.pdf"),
        )

    # Sorting curves
    for name in data.keys():

        if name == "test":
            col = "pred_conf" if n_splits == 1 else "ave_conf"
            compute_results_categorical(
                data[name],
                ["pred_conf1", "pred_conf2", "pred_conf3", "pred_conf4"],
                ["true_label"],
                np.expand_dims(data[name][col], 1),
                np.expand_dims(data[name]["aleatoric"], 1),
                np.expand_dims(data[name]["epistemic"], 1),
                fn=os.path.join(save_loc, "metrics"),
            )

        coverage_figures(
            data[name],
            output_features,
            colors=sym_colors,
            save_location=os.path.join(save_loc, "plots", f"coverage_{name}.pdf"),
        )
        
        if name == "test":

            # UQ figures
            uq_results(
                data[name], save_location=os.path.join(save_loc, "uq"), prefix=name
            )

    # Save metrics
    pd.DataFrame.from_dict(metrics).to_csv(
        os.path.join(save_loc, "metrics", "performance.csv")
    )


if __name__ == "__main__":

    description = "Usage: python evaluate_mlp.py -c model.yml"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)
    for newdir in ["plots", "uq", "metrics"]:
        os.makedirs(os.path.join(save_loc, newdir), exist_ok=True)

    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config_file, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    evaluate(conf)
