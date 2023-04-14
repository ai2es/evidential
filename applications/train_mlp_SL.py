import logging
import tqdm

from echo.src.base_objective import BaseObjective
import copy
import yaml
import shutil
import sys
import os
import gc
import optuna
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from argparse import ArgumentParser

from keras import backend as K
from evml.keras.models import RegressorDNN
from evml.keras.callbacks import get_callbacks
from evml.splitting import load_splitter
from evml.regression_uq import compute_results
from evml.preprocessing import load_preprocessing
from evml.keras.seed import seed_everything
from evml.pbs import launch_pbs_jobs


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):

        """Initialize the base class"""
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):
        K.clear_session()
        if "CSVLogger" in conf["callbacks"]:
            del conf["callbacks"]["CSVLogger"]
        if "ModelCheckpoint" in conf["callbacks"]:
            del conf["callbacks"]["ModelCheckpoint"]
        # Only use 1 data split
        conf["data"]["n_splits"] = 1

        try:
            return trainer(conf, trial=trial)
        except Exception as E:
            logger.warning(f"Trial {trial.number} failed due to error {str(E)}")
            raise optuna.TrialPruned()


def trainer(conf, trial=False):
    # load seed from the config and set globally
    seed = conf["seed"]
    seed_everything(seed)

    save_loc = conf["save_loc"]
    data_params = conf["data"]
    training_metric = conf["training_metric"]
    monte_carlo_passes = conf["monte_carlo_passes"]
    n_models = conf["n_models"]

    model_params = conf["model"]
    model_params["save_path"] = os.path.join(save_loc, "models")

    # Load data, splitter, and scalers
    data = pd.read_csv(conf["data"]["save_loc"])
    data["day"] = data["Time"].apply(lambda x: str(x).split(" ")[0])

    split_col = data_params["split_col"]
    n_splits = data_params["n_splits"]
    input_cols = data_params["input_cols"]
    output_cols = data_params["output_cols"]

    # Make some directories
    for super_dir in ["ensemble", "monte_carlo"]:  # seed
        os.makedirs(os.path.join(save_loc, super_dir), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{super_dir}/models"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{super_dir}/metrics"), exist_ok=True)
        os.makedirs(os.path.join(save_loc, f"{super_dir}/evaluate"), exist_ok=True)

    # Need the same test_data for all trained models (data and model ensembles)
    gsp = load_splitter(
        data_params["splitter"], n_splits=1, random_state=seed, train_size=0.9
    )
    splits = list(gsp.split(data, groups=data[split_col]))
    train_index, test_index = splits[0]
    _train_data, _test_data = (
        data.iloc[train_index].copy(),
        data.iloc[test_index].copy(),
    )

    # Save arrays for ensembles
    ensemble_mu = np.zeros((n_models, _test_data.shape[0], len(output_cols)))
    ensemble_sigma = np.zeros((n_models, _test_data.shape[0], len(output_cols)))
    
    best_model = None
    best_data_split = None
    best_model_score = 1e10

    for model_seed in range(n_models):

        # Make N train-valid splits using day as grouping variable
        gsp = load_splitter(
            data_params["splitter"],
            n_splits=n_splits,
            random_state=seed,
            train_size=0.885,
        )
        splits = list(gsp.split(_train_data, groups=_train_data[split_col]))

        # Train ensemble of parametric models
        _ensemble_pred = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))

        if n_models > 1: 
            # If only looping over data splits, the model is called below
            # Otherwise we call it here so it can be copied (same random seed)
            _model = RegressorDNN(**model_params)
            # build the model here so the weights are initialized (and can be copied below)
            _model.build_neural_network(_train_data[input_cols].values, _train_data[output_cols].values)

        # Create ensemble from n_splits number of data splits 
        for data_seed in range(n_splits):

            # select indices from the split, data splits
            train_index, valid_index = splits[data_seed]
            test_data = copy.deepcopy(_test_data)
            train_data, valid_data = (
                _train_data.iloc[train_index].copy(),
                _train_data.iloc[valid_index].copy(),
            )
            # preprocess x-transformations
            x_scaler, y_scaler = load_preprocessing(conf, seed=seed)
            if x_scaler:
                x_train = x_scaler.fit_transform(train_data[input_cols])
                x_valid = x_scaler.transform(valid_data[input_cols])
                x_test = x_scaler.transform(test_data[input_cols])
            else:
                x_train = train_data[input_cols].values
                x_valid = valid_data[input_cols].values
                x_test = test_data[input_cols].values

            # preprocess y-transformations
            if y_scaler:
                y_train = y_scaler.fit_transform(train_data[output_cols])
                y_valid = y_scaler.transform(valid_data[output_cols])
                y_test = y_scaler.transform(test_data[output_cols])
            else:
                y_train = train_data[output_cols].values
                y_valid = valid_data[output_cols].values
                y_test = test_data[output_cols].values

            # duplicate the model (same seed)
            model = RegressorDNN(**model_params)
            model.build_neural_network(x_train, y_train)
            if n_models > 1: # duplicate the model (same seed)
                model.model.set_weights(_model.model.get_weights())

            # callbacks
            if n_splits > 1:
                callbacks = get_callbacks(conf, path_extend="ensemble/models")
            else:
                callbacks = get_callbacks(conf, path_extend="monte_carlo/models")

            # fit the model
            model.fit(
                x_train,
                y_train,
                validation_data=(x_valid, y_valid),
                callbacks=callbacks,
            )
            history = model.model.history

            # If ECHO is running this script, n_splits has been set to 1, return the metric here
            if trial is not False:
                return {
                    x: min(y)
                    for x, y in history.history.items()
                    if x not in trial.params
                }

            # Save if its the best model
            if min(history.history[training_metric]) < best_model_score:
                best_model = model
                #best_data_split = data_seed
                model.model_name = "best.h5"
                model.save_model()

            # evaluate on the test holdout split
            _ensemble_pred[data_seed] = y_scaler.inverse_transform(
                model.predict(x_test)
            )

            # check if this is the best model
            del model
            tf.keras.backend.clear_session()
            gc.collect()

        if n_splits > 1:
            # Compute uncertainties for the data ensemble
            ensemble_mu[model_seed] = np.mean(_ensemble_pred, 0)
            ensemble_sigma[model_seed] = np.var(_ensemble_pred, 0)
        else:
            # Create ensemble from MC dropout
            dropout_mu = best_model.predict_monte_carlo(
                x_test, y_test, forward_passes=monte_carlo_passes, y_scaler=y_scaler
            )
            # Calculating mean across multiple MCD forward passes
            ensemble_mu[model_seed] = np.mean(dropout_mu, axis=0)  # shape (n_samples, n_classes)
            # Calculating variance across multiple MCD forward passes
            ensemble_sigma[model_seed] = np.var(dropout_mu, axis=0)  # shape (n_samples, n_classes)

    # Compute aleatoric and epistemic uncertainties using law of total uncertainty
    ensemble_mean = np.mean(ensemble_mu, 0)
    ensemble_aleatoric = np.mean(ensemble_sigma, 0)
    ensemble_epistemic = np.var(ensemble_mu, 0)

    # add to df and save
    _test_data[[f"{x}_ensemble_pred" for x in output_cols]] = ensemble_mean
    _test_data[[f"{x}_ensemble_ale" for x in output_cols]] = ensemble_aleatoric
    _test_data[[f"{x}_ensemble_epi" for x in output_cols]] = ensemble_epistemic

    # make some figures
    if n_splits > 1:
        _test_data.to_csv(os.path.join(save_loc, "ensemble/evaluate", "test.csv"))
        compute_results(
            _test_data,
            output_cols,
            ensemble_mean,
            ensemble_aleatoric,
            ensemble_epistemic,
            fn=os.path.join(save_loc, "ensemble/metrics"),
        )
    else:
        _test_data.to_csv(os.path.join(save_loc, "monte_carlo/evaluate", "test.csv"))
        compute_results(
            _test_data,
            output_cols,
            ensemble_mean,
            ensemble_aleatoric,
            ensemble_epistemic,
            fn=os.path.join(save_loc, "monte_carlo/metrics"),
        )


if __name__ == "__main__":

    description = "Train an MLP regression model on a surface layer data set"
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
        type=int,
        default=0,
        help="Submit 1 worker to PBS.",
    )
    args_dict = vars(parser.parse_args())
    config = args_dict.pop("model_config")
    launch = bool(int(args_dict.pop("launch")))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)

    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    if launch:
        from pathlib import Path

        script_path = Path(__file__).absolute()
        logging.info("Launching to PBS")
        launch_pbs_jobs(config, script_path)
        sys.exit()

    result = trainer(conf)