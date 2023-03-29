import logging
import tqdm

from echo.src.base_objective import BaseObjective
import copy
import yaml
import shutil
import sys
import os
import gc
import warnings
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import backend as K
from evml.keras.models import EvidentialRegressorDNN
from evml.keras.callbacks import get_callbacks
from evml.splitting import load_splitter
from evml.metrics import compute_results
from evml.preprocessing import load_preprocessing
from evml.keras.seed import seed_everything


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

    model_params = conf["model"]
    model_params["save_path"] = save_loc

    # Load data, splitter, and scalers
    data = pd.read_csv(conf["data"]["save_loc"])
    data["day"] = data["Time"].apply(lambda x: str(x).split(" ")[0])

    split_col = data_params["split_col"]
    n_splits = data_params["n_splits"]
    input_cols = data_params["input_cols"]
    output_cols = data_params["output_cols"]

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

    # Make N train-valid splits using day as grouping variable
    gsp = load_splitter(
        data_params["splitter"], n_splits=n_splits, random_state=seed, train_size=0.885
    )
    splits = list(gsp.split(_train_data, groups=_train_data[split_col]))

    # Train ensemble of parametric models
    ensemble_mu = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))
    ensemble_ale = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))
    ensemble_epi = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))

    best_model = None
    best_split = None
    best_model_score = 1e10
    for data_seed in tqdm.tqdm(range(n_splits)):
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

        # load the model
        model = EvidentialRegressorDNN(**model_params)
        model.build_neural_network(x_train, y_train)

        # Get callbacks
        callbacks = get_callbacks(conf, path_extend="")

        # fit model to training data
        history = model.model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            batch_size=model.batch_size,
            callbacks=callbacks,
            epochs=model.epochs,
            verbose=model.verbose,
            shuffle=True,
        )

        # If ECHO is running this script, n_splits has been set to 1, return the metric here
        if trial is not False:
            return {
                x: min(y) for x, y in history.history.items() if x not in trial.params
            }

        # Save if its the best model
        if min(history.history[training_metric]) < best_model_score:
            best_model = model
            model.model_name = "best.h5"
            model.save_model()
            best_split = data_seed

        # evaluate on the test holdout split
        mu, aleatoric, epistemic = model.predict(x_test)
        ensemble_mu[data_seed] = mu
        ensemble_ale[data_seed] = aleatoric
        ensemble_epi[data_seed] = epistemic

        # check if this is the best model
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # Compute uncertainties
    mu = ensemble_mu[best_split]
    epistemic = ensemble_epi[best_split]
    aleatoric = ensemble_ale[best_split]

    # add to df and save
    _test_data[[f"{x}_pred" for x in output_cols]] = mu
    _test_data[[f"{x}_ale" for x in output_cols]] = aleatoric
    _test_data[[f"{x}_epi" for x in output_cols]] = epistemic
    _test_data.to_csv(os.path.join(save_loc, "test.csv"))

    # make some figures
    os.makedirs(os.path.join(save_loc, "evidential"), exist_ok=True)
    compute_results(
        _test_data,
        output_cols,
        mu,
        aleatoric,
        epistemic,
        fn=os.path.join(save_loc, "evidential"),
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python train_SL.py model.yml")
        sys.exit()

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)

    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    result = trainer(conf)
