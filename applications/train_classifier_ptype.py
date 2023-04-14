import logging
from echo.src.base_objective import BaseObjective
from echo.src.trial_suggest import trial_suggest_loader
import yaml
import shutil
import os
import gc
import tqdm
import optuna
import pickle
import warnings
import numpy as np
import pandas as pd 
from tensorflow.keras import backend as K
from argparse import ArgumentParser

from ptype.callbacks import MetricsCallback
from ptype.data import load_ptype_data_day, preprocess_data

from evml.keras.callbacks import get_callbacks, ReportEpoch
from evml.keras.models import CategoricalDNN
from evml.pbs import launch_pbs_jobs
from bridgescaler import save_scaler


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):

        """Initialize the base class"""
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):
        K.clear_session()
        gc.collect()
        conf["n_splits"] = 1
        if "CSVLogger" in conf["callbacks"]:
            del conf["callbacks"]["CSVLogger"]
        if "ModelCheckpoint" in conf["callbacks"]:
            del conf["callbacks"]["ModelCheckpoint"]
        if "rain_weight" in conf["optuna"]["parameters"]:
            conf = self.custom_updates(trial, conf)
        try:
            return {self.metric: trainer(conf, evaluate=False)}
        except Exception as E:
            if "Unexpected result" in str(E) or "CUDA" in str(E):
                logger.warning(
                    f"Pruning trial {trial.number} due to unspecified error: {str(E)}."
                )
                raise optuna.TrialPruned()
            else:
                logger.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E

    def custom_updates(self, trial, conf):
        # Get list of hyperparameters from the config
        hyperparameters = conf["optuna"]["parameters"]
        # Now update some via custom rules
        weights = []
        for ptype in ["rain_weight", "snow_weight", "sleet_weight", "frz_rain_weight"]:
            value = trial_suggest_loader(trial, hyperparameters[ptype])
            logger.info(f"Updated {ptype} with value {value}")
            weights.append(value)
        # Update the config based on optuna's suggestion
        conf["model"]["loss_weights"] = weights
        return conf


def trainer(conf, evaluate=True, data_split=0, mc_forward_passes = 0):
    input_features = (
        conf["TEMP_C"] + conf["T_DEWPOINT_C"] + conf["UGRD_m/s"] + conf["VGRD_m/s"]
    )
    output_features = conf["ptypes"]
    metric = conf["metric"]
    # flag for using the evidential model
    if conf["model"]["loss"] == "dirichlet":
        use_uncertainty = True
    else:
        use_uncertainty = False
    # load data using the split (see n_splits in config)
    data = load_ptype_data_day(conf, data_split=data_split, verbose=1, drop_mixed = False)
    # check if we should scale the input data by groups
    scale_groups = [] if "scale_groups" not in conf else conf["scale_groups"]
    groups = [conf[g] for g in scale_groups]
    leftovers = list(
        set(input_features)
        - set([row for group in scale_groups for row in conf[group]])
    )
    if len(leftovers):
        groups.append(leftovers)
    # scale the data
    scaled_data, scalers = preprocess_data(
        data,
        input_features,
        output_features,
        scaler_type="standard",
        encoder_type="onehot",
        groups=groups,
    )
    # Save the scalers when not using ECHO
    if evaluate:
        os.makedirs(os.path.join(conf["save_loc"], "scalers"), exist_ok=True)
        for scaler_name, scaler in scalers.items():
            if conf["n_splits"] == 1:
                fn = os.path.join(conf["save_loc"], "scalers", f"{scaler_name}.json")
            else:
                fn = os.path.join(conf["save_loc"], "scalers", f"{scaler_name}_{data_split}.json")
            try:
                save_scaler(scaler, fn)
            except TypeError:
                with open(fn, "wb") as fid:
                    pickle.dump(scaler, fid)
    # set up callbacks
    callbacks = []
    if use_uncertainty:
        callbacks.append(ReportEpoch(conf["model"]["annealing_coeff"]))
    if "ModelCheckpoint" in conf["callbacks"]:  # speed up echo
        callbacks.append(
            MetricsCallback(
                scaled_data["train_x"],
                scaled_data["train_y"],
                name="train",
                use_uncertainty=use_uncertainty,
            )
        )
        callbacks.append(
            MetricsCallback(
                scaled_data["test_x"],
                scaled_data["test_y"],
                name="test",
                use_uncertainty=use_uncertainty,
            )
        )
    callbacks.append(
        MetricsCallback(
            scaled_data["val_x"],
            scaled_data["val_y"],
            name="val",
            use_uncertainty=use_uncertainty,
        )
    )
    callbacks += get_callbacks(conf, path_extend="models")
    # initialize the model
    mlp = CategoricalDNN(**conf["model"], callbacks=callbacks)
    # train the model
    history = mlp.fit(scaled_data["train_x"], scaled_data["train_y"])
    
    if conf["n_splits"] > 1:
        pd_history = pd.DataFrame.from_dict(history.history)
        pd_history["split"] = data_split
        pd_history.to_csv(os.path.join(conf["save_loc"], "models", f"training_log_{data_split}.csv"))
    
    # Predict on the data splits
    if evaluate:
        # Save the best model when not using ECHO
        if conf["n_splits"] == 1:
            mlp.model.save(os.path.join(conf["save_loc"], "models", "model.h5"))
        else:
            mlp.model.save(os.path.join(conf["save_loc"], "models", f"model_{data_split}.h5"))
        for name in data.keys():
            x = scaled_data[f"{name}_x"]
            pred_probs = mlp.predict(x)
            if use_uncertainty:
                pred_probs, u, ale, epi = mlp.compute_uncertainties(pred_probs)
                pred_probs = pred_probs.numpy()
                u = u.numpy()
                ale = ale.numpy()
                epi = epi.numpy()
            elif mc_forward_passes > 0:  # Compute epistemic uncertainty with MC dropout
                pred_probs = mlp.predict(x)
                _, epi, entropy, mutual_info = mlp.predict_monte_carlo(
                    x, mc_forward_passes=mc_forward_passes
                )
            true_labels = np.argmax(data[name][output_features].to_numpy(), 1)
            pred_labels = np.argmax(pred_probs, 1)
            confidences = np.take_along_axis(pred_probs, pred_labels[:, None], axis=1)
            data[name]["true_label"] = true_labels
            data[name]["pred_label"] = pred_labels
            data[name]["pred_conf"] = confidences
            for k in range(pred_probs.shape[-1]):
                data[name][f"pred_conf{k+1}"] = pred_probs[:, k]
            if use_uncertainty:
                data[name]["evidential"] = u
                data[name]["aleatoric"] = np.take_along_axis(
                    ale, pred_labels[:, None], axis=1
                )
                data[name]["epistemic"] = np.take_along_axis(
                    epi, pred_labels[:, None], axis=1
                )
            elif mc_forward_passes > 0:
                data[name]["epistemic"] = np.take_along_axis(
                    epi, pred_labels[:, None], axis=1
                )
                data[name]["entropy"] = entropy
                data[name]["mutual_info"] = mutual_info
                
            if conf["n_splits"] == 1:
                data[name].to_parquet(os.path.join(conf["save_loc"], f"evaluate/{name}.parquet"))
            else:
                data[name].to_parquet(os.path.join(conf["save_loc"], f"evaluate/{name}_{data_split}.parquet"))
        return 1

    elif conf["direction"] == "max":  # Return metric to be used in ECHO
        return max(history.history[metric])
    else:
        return min(history.history[metric])


if __name__ == "__main__":

    description = "Usage: python train_mlp.py -c model.yml"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=str,
        default=False,
        help="Launch n_splits number of qsub jobs.",
    )
    parser.add_argument(
        "-s",
        dest="serial",
        type=str,
        default=False,
        help="Whether to parallelize the training over GPUs (default is 0)",
    )
    parser.add_argument(
        "-n",
        dest="n_splits",
        type=int,
        default=1,
        help="The number of data ensembles to create (n_splits). Default = 1",
    )
    parser.add_argument(
        "-i",
        dest="split_id",
        type=int,
        default=0,
        help="Which split this node will run (ranges from 0 to n_splits-1)",
    )
    parser.add_argument(
        "-m",
        dest="mc_steps",
        type=int,
        default=0,
        help="The number of MC steps used to create an ensemble (default = 0)",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")
    
    launch = bool(int(args_dict.pop("launch")))
    n_splits = int(args_dict.pop("n_splits"))
    this_split = int(args_dict.pop("split_id"))
    mc_steps = int(args_dict.pop("mc_steps"))
    run_serially = bool(int(args_dict.pop("serial")))

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        
    # If we are running the training and not launching
    #conf["n_splits"] = n_splits

    # Create the save directory if does not exist
    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)
    os.makedirs(os.path.join(save_loc, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_loc, "evaluate"), exist_ok=True)

    # Copy the model config file to the new directory
    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config_file, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)
    
    if launch:
        if run_serially:
            # If we are running serially, launch only one job
            # set serial flag = True
            # set launch flag = False
                pass
        else:
            # Launch QSUB jobs and exit
            for split in range(n_splits):
                   pass 
        sys.exit()
    
    # Run in serial over the number of ensembles (one at a time)
    if run_serially:
        for split in tqdm.tqdm(range(n_splits)):
            trainer(conf, 
                    data_split = split, 
                    mc_forward_passes = mc_steps)
    
    # Run one ensemble
    else:
        trainer(conf, 
                data_split = this_split, 
                mc_forward_passes = mc_steps)
