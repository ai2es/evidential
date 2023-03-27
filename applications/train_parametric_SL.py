import logging
import tqdm

from echo.src.base_objective import BaseObjective
import copy
import yaml
import shutil
import sys
import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from evml.keras.models import ParametricRegressorDNN
from evml.keras.callbacks import get_callbacks
from evml.splitting import load_splitter
from evml.keras.monte_carlo import monte_carlo_ensemble
from evml.metrics import compute_results


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def seed_everything(seed=1234):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


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
        return trainer(conf, save=False)


def trainer(conf, evaluate=True, trial = False):

    seed = conf["seed"]
    save_loc = conf["save_loc"]
    data_params = conf["data"]
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
    ensemble_var = np.zeros((n_splits, _test_data.shape[0], len(output_cols)))

    best_model = None
    best_model_score = 1e10
    for data_seed in tqdm.tqdm(range(n_splits)):
        train_index, valid_index = splits[data_seed]
        test_data = copy.deepcopy(_test_data)
        train_data, valid_data = (
            _train_data.iloc[train_index].copy(),
            _train_data.iloc[valid_index].copy(),
        )

        x_scaler, y_scaler = RobustScaler(), MinMaxScaler((0, 1))
        x_train = x_scaler.fit_transform(train_data[input_cols])
        x_valid = x_scaler.transform(valid_data[input_cols])
        x_test = x_scaler.transform(test_data[input_cols])

        y_train = y_scaler.fit_transform(train_data[output_cols])
        y_valid = y_scaler.transform(valid_data[output_cols])
        y_test = y_scaler.transform(test_data[output_cols])

        model = ParametricRegressorDNN(**model_params)
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
        if trial != False:
            return {x: min(x) for x in history.history}

        # Save if its the best model
        if min(history.history[f'val_{model_params["metrics"]}']) < best_model_score:
            best_model = model
            model.model_name = "best.h5"
            model.save_model()

        # evaluate on the test holdout split
        y_pred = model.predict(x_test)
        mu, aleatoric = model.calc_uncertainties(y_pred, y_scaler)
        if mu.shape[-1] == 1:
            mu = np.expand_dims(mu)
            aleatoric = np.expand_dims(aleatoric, 1)
        ensemble_mu[data_seed] = mu
        ensemble_var[data_seed] = aleatoric

        # check if this is the best model
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # Compute uncertainties
    ensemble_mu = np.mean(ensemble_mu, axis=0)
    ensemble_epistemic = np.var(ensemble_mu, axis=0)
    ensemble_aleatoric = np.mean(ensemble_var, axis=0)

    # Compute epistemic uncertainty via MC-dropout
    mc_mu, mc_aleatoric = monte_carlo_ensemble(
        best_model, x_test, y_test, forward_passes=5, y_scaler=y_scaler
    )
    mc_epistemic = np.var(mc_mu)
    mc_aleatoric = np.mean(mc_aleatoric)

    # add to df and save
    _test_data[[f"{x}_ensemble_pred" for x in output_cols]] = ensemble_mu
    _test_data[[f"{x}_mc_pred" for x in output_cols]] = mc_mu[:, :, 0]
    _test_data[[f"{x}_ensemble_ale" for x in output_cols]] = ensemble_aleatoric
    _test_data[[f"{x}_mc_ale" for x in output_cols]] = mc_aleatoric
    _test_data[[f"{x}_ensemble_epi" for x in output_cols]] = ensemble_epistemic
    _test_data[[f"{x}_mc_epi" for x in output_cols]] = mc_epistemic
    _test_data.to_csv(os.path.join(save_loc, "test.csv"))

    # make some figures
    os.mkdirs(os.path.join(save_loc, "ensemble"), exist_ok=True)
    compute_results(
        _test_data,
        output_cols,
        ensemble_mu,
        ensemble_aleatoric,
        ensemble_epistemic,
        fn=os.path.join(save_loc, "ensemble"),
    )
    os.mkdirs(os.path.join(save_loc, "ensemble"), exist_ok=True)
    compute_results(
        _test_data,
        output_cols,
        mc_mu[:, :, 0],
        mc_aleatoric,
        mc_epistemic,
        fn=os.path.join(save_loc, "monte_carlo"),
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python train_SL.py model.yml")
        sys.exit()

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

    ave_csi = trainer(conf)
    print(ave_csi)
