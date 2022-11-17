import logging, tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from echo.src.base_objective import BaseObjective
import copy, joblib
import time
import yaml
import shutil
import sys
import os
import gc
import sklearn
import random
import warnings 
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from keras import backend as K
from collections import OrderedDict
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from evml.reliability import compute_calibration, reliability_diagram, reliability_diagrams
from evml.plotting import plot_confusion_matrix, conus_plot
from evml.keras.models import build_model, calc_prob_uncertinty
from evml.keras.losses import Dirichlet

import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
from functools import partial

from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
from hagelslag.evaluation.MetricPlotter import roc_curve, performance_diagram

from imblearn.under_sampling import RandomUnderSampler
from imblearn.tensorflow import balanced_batch_generator

from cartopy import crs as ccrs
from cartopy import feature as cfeature
warnings.filterwarnings("ignore")



logger = logging.getLogger(__name__)


def seed_everything(seed=1234):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
            
        return trainer(conf, save = False)
    
    
class CSICallback(tf.keras.callbacks.Callback):
    def __init__(self, model, x_test, y_test):
        super(CSICallback, self).__init__()
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.x_test))
        logs["val_csi"] = mean_csi(self.y_test, y_pred)
        return
    

def mean_csi(y_true, y_pred):
    pred_probs, u = calc_prob_uncertinty(y_pred)
    pred_probs = pred_probs.numpy()
    u = u.numpy()
    #true_labels = np.argmax(y_true, 1) 
    pred_labels = np.argmax(pred_probs, 1)
    confidences = np.take_along_axis(
        pred_probs,
        pred_labels[:, None], 
        axis=1
    )
    rocs = []
    for i in range(pred_probs.shape[1]):
        forecasts = confidences.copy()
        obs = np.where(np.argmax(y_true, 1)  == i, 1, 0)
        roc = DistributedROC(thresholds=np.arange(0.0, 1.01, 0.01),
                             obs_threshold=0.5)
        roc.update(forecasts[:, 0], obs)
        rocs.append(roc.max_csi())
    return np.mean(rocs)

        
class ReportEpoch(tf.keras.callbacks.Callback):
    def __init__(self, annealing_coeff):
        super(ReportEpoch, self).__init__()
        self.this_epoch = 0
        self.annealing_coeff = annealing_coeff
    
    def on_epoch_begin(self, epoch, logs={}):
        self.this_epoch += 1
        


def trainer(conf, evaluate = True, data_seed = 0):
    features = conf['tempvars'] + conf['tempdewvars'] + conf['ugrdvars'] + conf['vgrdvars']
    outputs = conf['outputvars']
    num_classes = len(outputs)
    n_splits = conf['trainer']['n_splits']
    train_size1 = conf['trainer']['train_size1'] # sets test size
    train_size2 = conf['trainer']['train_size2'] # sets valid size
    num_hidden_layers = conf['trainer']['num_hidden_layers']
    middle_size = conf['trainer']['hidden_sizes']
    dropout_rate = conf['trainer']['dropout_rate']
    batch_norm = conf['trainer']['batch_norm']
    batch_size = conf['trainer']['batch_size']
    balance_batches = conf['trainer']['balance_batches']
    learning_rate = conf['trainer']['learning_rate']
    label_smoothing = conf["trainer"]["label_smoothing"]
    outputvar_weights = conf["trainer"]["outputvar_weights"]
    L2_reg = conf['trainer']['l2_reg']
    metrics = conf['trainer']['metrics']
    run_eagerly = conf['trainer']['run_eagerly']
    shuffle = conf['trainer']['shuffle']
    epochs = conf['trainer']['epochs']
    seed = conf["seed"]
    verbose = conf["verbose"]
    lr_patience = conf["trainer"]["lr_patience"]
    stopping_patience = conf["trainer"]["stopping_patience"]
    loss = conf["trainer"]["loss"]
    use_uncertainty = False if loss == "ce" else True
    
    ### Set the seed for reproducibility
    seed_everything(seed)

    ### Load the data
    if not os.path.isfile(os.path.join(conf['data_path'], "cached.parquet")):
        df = pd.concat([
            pd.read_parquet(x) for x in tqdm.tqdm(glob.glob(os.path.join(conf['data_path'], "*.parquet")))
        ])
        df.to_parquet(os.path.join(conf['data_path'], "cached.parquet"))
    else:
        df = pd.read_parquet(os.path.join(conf['data_path'], "cached.parquet"))

    ### Split and preprocess the data
    df['day'] = df['datetime'].apply(lambda x: str(x).split(' ')[0])
    df["id"] = range(df.shape[0])

    texas = [f"2021-02-{k}" for k in range(10, 20)] #texas
    new_york = ["2022-02-03", "2022-02-04"] # New York
    ne_noreaster = [f"2017-03-{k}" for k in range(11, 18)] # NE NorEaster
    dec_ice_storm = [f"2016-12-{k}" for k in range(15, 21)]
    test_days = texas + new_york + ne_noreaster + dec_ice_storm
    test_days_c = df["day"].isin(test_days)
    
    # Need the same test_data for all trained models (data and model ensembles)
    #n_splits = 10
    flat_seed = conf["seed"]
    gsp = GroupShuffleSplit(n_splits=conf["trainer"]["n_splits"],
                            random_state = flat_seed, 
                            train_size=conf["trainer"]["train_size1"])
    splits = list(gsp.split(df[~test_days_c], groups = df[~test_days_c]["day"]))
    train_index, test_index = splits[0]
    train_data, test_data = df[~test_days_c].iloc[train_index].copy(), df[~test_days_c].iloc[test_index].copy() 
    test_data = pd.concat([test_data, df[test_days_c].copy()])

    # Make N train-valid splits using day as grouping variable
    gsp = GroupShuffleSplit(n_splits=conf["trainer"]["n_splits"],
                            random_state = flat_seed, 
                            train_size=conf["trainer"]["train_size2"])
    splits = list(gsp.split(train_data, groups = train_data["day"]))

    train_index, valid_index = splits[data_seed]
    train_data, valid_data = train_data.iloc[train_index].copy(), train_data.iloc[valid_index].copy() 

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(train_data[features])
    x_valid = scaler_x.transform(valid_data[features])
    x_test = scaler_x.transform(test_data[features])
    y_train = tf.keras.utils.to_categorical(np.argmax(train_data[outputs].to_numpy(), 1), num_classes)
    y_valid = tf.keras.utils.to_categorical(np.argmax(valid_data[outputs].to_numpy(), 1), num_classes)
    y_test = tf.keras.utils.to_categorical(np.argmax(test_data[outputs].to_numpy(), 1), num_classes)

    if balance_batches:
        train_idx = np.argmax(train_data[outputs].to_numpy(), 1)
        weights = np.array(conf["trainer"]["outputvar_weights"])
        training_generator, steps_per_epoch = balanced_batch_generator(
            x_train,
            y_train,
            sample_weight=np.array([weights[l] for l in train_idx]),
            sampler=RandomUnderSampler(),
            batch_size=conf["trainer"]["batch_size"],
            random_state=flat_seed,
        )
    else:
        if conf['trainer']['use_weights']:
            class_weight = {k: v for k, v in enumerate(conf["trainer"]["outputvar_weights"])}
        else:
            class_weight = None
    
    mlp = build_model(
        len(features),
        conf["trainer"]["hidden_sizes"],
        conf["trainer"]["num_hidden_layers"],
        len(outputs), 
        conf["trainer"]["activation"], 
        conf["trainer"]["dropout_rate"],
        conf["trainer"]["out_activation"])

    mlp.build((conf["trainer"]["batch_size"], len(features)))
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=conf["trainer"]["learning_rate"]
    )
    
    annealing_coeff = 50
    epoch_callback = ReportEpoch(annealing_coeff)
    # load loss 
    if use_uncertainty:
        criterion = partial(
            Dirichlet,
            callback = epoch_callback, 
            weights = False
        )
    else:
        criterion = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            name='categorical_crossentropy'
        )
    # compile model
    mlp.compile(
        optimizer = optimizer, 
        loss = criterion,
        metrics = ["accuracy",
                   tf.keras.metrics.Precision(name="prec"),
                   tf.keras.metrics.Recall(name="recall"),
                   tfa.metrics.F1Score(num_classes=num_classes,
                                       average='macro', name="f1"),
                   tf.keras.metrics.AUC(name = "auc")]
    )
    # create early stopping callback
    callback1 = tf.keras.callbacks.EarlyStopping(
        monitor = metrics, 
        mode = 'max',
        patience = 15,
        restore_best_weights = True
    )

    # create reduce LR callback
    callback2 = tf.keras.callbacks.ReduceLROnPlateau(
        monitor = metrics, 
        factor = 0.1, 
        patience = 3, 
        verbose = 0,
        mode = 'max',
        min_lr = 1e-7
    )
    # if evaluate: add save callbacks for the training log and the model weights
    
    # average CSI, others
    csi_callback = CSICallback(
        mlp, x_valid, y_valid)
    
    if balance_batches:
        history = mlp.fit(
            training_generator, 
            steps_per_epoch=steps_per_epoch,
            validation_data = (x_valid, y_valid),
            epochs = conf["trainer"]["epochs"],
            callbacks = [csi_callback, epoch_callback, callback1, callback2],
            verbose = 2,
            shuffle = True
        )
    else:   
        history = mlp.fit(
            x=x_train, 
            y=y_train, 
            validation_data = (x_valid, y_valid),
            class_weight = class_weight,
            epochs = conf["trainer"]["epochs"],
            batch_size = conf["trainer"]["batch_size"],
            callbacks = [predict_callback, epoch_callback, callback1, callback2],
            verbose = 2,
            shuffle = True
        )
    
    if evaluate:
        names = ["train", "valid", "test"]
        splits = [x_train, x_valid, x_test]
        dfs = [train_data, valid_data, test_data]
        for name, x, df in zip(names, splits, dfs):
            y_pred = mlp.predict(x)
            if use_uncertainty:
                pred_probs, u = calc_prob_uncertinty(y_pred)
                pred_probs = pred_probs.numpy()
                u = u.numpy()
            else:
                pred_probs = y_pred
            true_labels = np.argmax(df[outputs].to_numpy(), 1)
            pred_labels = np.argmax(pred_probs, 1)
            confidences = np.take_along_axis(
                pred_probs,
                pred_labels[:, None], 
                axis=1
            )
            df["true_label"] = true_labels
            df["pred_label"] = pred_labels
            df["pred_conf"] = confidences
            for k in range(pred_probs.shape[-1]):
                df[f"pred_conf{k+1}"] = pred_probs[:, k]
            if use_uncertainty:
                df["pred_sigma"] = u
            df.to_parquet(
                os.path.join(conf["save_loc"], f"{name}_{data_seed}.parquet"))
        return 1
            
    else: # Return metric to be used in ECHO 
        y_pred = mlp.predict(x_test)
        if use_uncertainty:
            pred_probs, u = calc_prob_uncertinty(y_pred)
            pred_probs = pred_probs.numpy()
            u = u.numpy()
        else:
            pred_probs = y_pred

        true_labels = np.argmax(test_data[outputs].to_numpy(), 1)
        pred_labels = np.argmax(pred_probs, 1)

        confidences = np.take_along_axis(
            pred_probs,
            pred_labels[:, None], 
            axis=1
        ) 
        test_data["true_label"] = true_labels
        test_data["pred_label"] = pred_labels
        test_data["pred_conf"] = confidences
        for k in range(pred_probs.shape[-1]):
            test_data[f"pred_conf{k+1}"] = pred_probs[:, k]
        if use_uncertainty:
            test_data["pred_sigma"] = u

        rocs = []
        for i in range(num_classes):
            forecasts = test_data['pred_conf']
            obs = np.where(test_data['true_label'] == i, 1, 0)
            roc = DistributedROC(thresholds=np.arange(0.0, 1.1, 0.1),
                                 obs_threshold=0.5)
            roc.update(forecasts.astype(float), obs.astype(float))
            rocs.append(roc.max_csi())

        return np.mean(rocs)
    


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python train_classifier_keras.py model.yml")
        sys.exit()

    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        
    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok = True)
    
    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)
            
    ave_csi = trainer(conf)
    print(ave_csi)


#     dfs = [train_data, valid_data, test_data]
#     loaders = [train_loader, valid_loader, test_loader]
#     save_names = ["train", "valid", "test"]
#     for df, loader, name in zip(dfs, loaders, save_names):
    
#         results = validate(
#                 epoch,
#                 model,
#                 loader,
#                 num_classes,
#                 criterion,
#                 batch_size,
#                 device=device,
#                 uncertainty=use_uncertainty,
#                 return_preds=True
#             )

#         ### Add the predictions to the original dataframe and save to disk
#         for idx in range(len(outputs)):
#             df[f"{outputs[idx]}_conf"] = results["pred_probs"][:, idx].cpu().numpy()
#         if use_uncertainty:
#             df["uncertainty"] = results["pred_uncertainty"][:, 0].cpu().numpy()
#         df["pred_labels"] = results["pred_labels"][:, 0].cpu().numpy()
#         df["true_labels"] = results["true_labels"][:, 0].cpu().numpy()
#         df["pred_conf"] = np.max(results["pred_probs"].cpu().numpy(), 1)
#         df.to_parquet(f"{save_loc}/{name}.parquet")