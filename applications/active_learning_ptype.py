import logging, tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader

import copy, joblib
import time, yaml
import torch.nn.functional as F
import pandas as pd, numpy as np
pd.options.mode.chained_assignment = None

from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

from evml.reliability import reliability_diagram, reliability_diagrams, compute_calibration
from evml.class_losses import *
from evml.model import seed_everything, DNN
from evml.mc_dropout import monte_carlo_dropout

from evml.training import train_one_epoch, validate
import random, os, numpy as np, sys, shutil, glob
from functools import partial


def train(conf, data, train_metric = "valid_auc", direction = "max", mc_forward_passes = 50):
    
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
    policy = conf["trainer"]["policy"]

    lr_patience = conf["trainer"]["lr_patience"]
    stopping_patience = conf["trainer"]["stopping_patience"]

    loss = conf["trainer"]["loss"]
    use_uncertainty = False if loss == "ce" else True
    
    ### Set the seed for reproducibility
    seed_everything(seed)

    ### Load the data
    train_data = data["train_data"]
    valid_data = data["valid_data"]
    test_data = data["test_data"]
    left_overs = data["left_overs"]
    
    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(train_data[features])
    x_valid = scaler_x.transform(valid_data[features])
    x_test = scaler_x.transform(test_data[features])
    x_left = scaler_x.transform(left_overs[features])
    y_train = np.argmax(train_data[outputs].to_numpy(), 1)
    y_valid = np.argmax(valid_data[outputs].to_numpy(), 1)
    y_test = np.argmax(test_data[outputs].to_numpy(), 1)
    y_left = np.argmax(left_overs[outputs].to_numpy(), 1)

    ### Use torch wrappers for convenience
    train_split = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long()
    )
    ### Small dataset so we dont need more than 1 worker
    train_loader = DataLoader(train_split, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=0)

    valid_split = TensorDataset(
        torch.from_numpy(x_valid).float(),
        torch.from_numpy(y_valid).long()
    )
    valid_loader = DataLoader(valid_split, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=0)
    
    test_split = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_split, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=0)
    
    left_split = TensorDataset(
        torch.from_numpy(x_left).float(),
        torch.from_numpy(y_left).long()
    )
    left_loader = DataLoader(left_split, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=0)

    ### Set the GPU device(s)
    device = get_device()
    
    ### Set up the loss
    weights = torch.from_numpy(np.array(outputvar_weights)).float().to(device)
    weights /= weights.sum()
    if use_uncertainty:
        if loss == "digamma":
            criterion = edl_digamma_loss
        elif loss == "log":
            criterion = edl_log_loss
        elif loss == "mse":
            criterion = edl_mse_loss
        else:
            logging.error("--uncertainty requires --mse, --log or --digamma.")
    else:
        criterion = nn.CrossEntropyLoss(
            weight = weights,
            label_smoothing = label_smoothing
        ).to(device)

    ### Load MLP model
    model = DNN(
            len(features), 
            len(outputs), 
            block_sizes = [middle_size for _ in range(num_hidden_layers)], 
            dr = [dropout_rate for _ in range(num_hidden_layers)], 
            batch_norm = batch_norm, 
            lng = False
        ).to(device)
    
    ### Initialize an optimizer
    optimizer = optim.Adam(model.parameters(), 
                           lr=learning_rate, 
                           weight_decay=L2_reg)

    ### Load a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience = lr_patience, 
        verbose = False,
        min_lr = 1.0e-13
    )
    
    ### Train the model
    best_model = False
    results_dict = defaultdict(list)
    for epoch in range(epochs):
        ### Train one epoch
        train_results, model, optimizer = train_one_epoch(
            epoch,
            model,
            train_loader,
            num_classes,
            criterion,
            optimizer,
            batch_size,
            weights=weights,
            device=device,
            uncertainty=use_uncertainty,
            verbose=False
        )
        ### Validate one epoch
        valid_results = validate(
            epoch,
            model,
            valid_loader,
            num_classes,
            criterion,
            batch_size,
            weights=weights,
            device=device,
            uncertainty=use_uncertainty,
            return_preds=False,
            verbose=False
        )
        
        ## Add train / valid metrics to the main results df
        for metric in train_results.keys():
            results_dict[f"train_{metric}"].append(np.mean(train_results[metric]))
            results_dict[f"valid_{metric}"].append(np.mean(valid_results[metric]))
        
        ### Save the dataframe to disk
        df = pd.DataFrame.from_dict(results_dict).reset_index()
        #df.to_csv(f"{save_loc}/training_log.csv", index = False)
        
        ### Find the best value so far
        if direction == "max":
            best_value = max(results_dict[train_metric])
            annealing_value = 1 - best_value
        else:
            best_value = min(results_dict[train_metric])
            annealing_value = best_value
        
        ### Save the model only if its the best
        if results_dict[train_metric][-1] == best_value:
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_x': scaler_x,
                train_metric: best_value
            }
            #torch.save(state_dict, f"{save_loc}/best.pt")
            best_model = model
            best_epoch = epoch
        
        ### Update the scheduler
        lr_scheduler.step(annealing_value)
        
        ### Early stopping
        best_epoch = [i for i,j in enumerate(
            results_dict[train_metric]) if j == best_value][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break
            
    ### Evaluate with the best model    
    ### Predict on the splits
    train_results = validate(
            best_epoch,
            best_model,
            train_loader,
            num_classes,
            criterion,
            batch_size,
            device=device,
            uncertainty=use_uncertainty,
            return_preds=False,
            verbose=False
        )
    
    valid_results = validate(
            best_epoch,
            best_model,
            valid_loader,
            num_classes,
            criterion,
            batch_size,
            device=device,
            uncertainty=use_uncertainty,
            return_preds=False,
            verbose=False
        )
    
    test_results = validate(
            best_epoch,
            best_model,
            test_loader,
            num_classes,
            criterion,
            batch_size,
            device=device,
            uncertainty=use_uncertainty,
            return_preds=False,
            verbose=False
        )

    ### Add the predictions to the original dataframe and save to disk
    #for idx in range(len(outputs)):
    #    test_data[f"{outputs[idx]}_conf"] = results["pred_probs"][:, idx].cpu().numpy()
    #if use_uncertainty:
    #    test_data["uncertainty"] = results["pred_uncertainty"][:, 0].cpu().numpy()
    #test_data["pred_labels"] = results["pred_labels"][:, 0].cpu().numpy()
    #test_data["true_labels"] = results["true_labels"][:, 0].cpu().numpy()
    #test_data["pred_conf"] = np.max(results["pred_probs"].cpu().numpy(), 1)
    
    ### Predict on the leftover data
    if policy == "mc_dropout":
        n_samples = x_left.shape[0]
        mc_results = monte_carlo_dropout(
            left_loader,
            mc_forward_passes,
            best_model,
            num_classes,
            n_samples, 
            batch_size=batch_size, 
            uncertainty=use_uncertainty
        )
        # take the top prediction and computed variance
        left_overs["uncertainty"] = np.take_along_axis(
            mc_results["variance"],
            np.argmax(mc_results["mean"], axis = 1)[:, None], 
            axis=1
        )
    elif policy == "evidential":
        leftover_results = validate(
                best_epoch,
                best_model,
                left_loader,
                num_classes,
                criterion,
                batch_size,
                device=device,
                uncertainty=use_uncertainty,
                return_preds=True,
                verbose=False
            )
        ### Add the predictions to the original dataframe and save to disk
        left_overs["uncertainty"] = leftover_results["pred_uncertainty"][:, 0].cpu().numpy()
        
    return df, train_results, valid_results, test_results, left_overs


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python train_classifier.py model.yml")
        sys.exit()

    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        
    start_at_split = 0
    if len(sys.argv) == 3:
        start_at_split = int(sys.argv[2])
        
    save_loc = conf["save_loc"]
    seed = conf["seed"]
    os.makedirs(save_loc, exist_ok = True)
    
    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)
            
    #seed_everything(seed)
    num_iterations = conf["trainer"]["num_iterations"]
    policy = conf["trainer"]["policy"]

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
    train_metric = "valid_auc"
    direction = "max"
    
    splitter = GroupShuffleSplit(n_splits=conf['trainer']['n_splits'], 
                                 train_size=conf['trainer']['train_size1'], 
                                 random_state = seed)
    train_idx, test_idx = list(splitter.split(df, groups=df['day']))[0]
    train_data, test_data = df.iloc[train_idx], df.iloc[test_idx]
    num_selected = int((1. / num_iterations) * train_data.shape[0])
    
    for sidx in range(conf['trainer']['n_splits']):
        
        if sidx < start_at_split:
            continue
    
        my_iter = tqdm.tqdm(
            range(num_iterations), 
            total = num_iterations, 
            leave = True
        )
        
        active_results = defaultdict(list)
        for iteration in my_iter:

            if iteration == 0:
                ### Select random fraction on first pass
                train_data_ = train_data.sample(n = num_selected, random_state = seed)
                left_overs = np.array(list(set(train_data["id"]) - set(train_data_["id"])))
                left_overs = train_data[train_data["id"].isin(left_overs)].copy()
            else:
                ### Select with a policy
                if policy == "random":
                    selection = left_overs.sample(n = num_selected, random_state = seed)
                else:
                    left_overs = left_overs.sort_values("uncertainty", ascending = False)
                    if num_selected > left_overs.shape[0]:
                        selection = left_overs.copy()
                    else:
                        selection = left_overs.iloc[:num_selected].copy()
                train_data_ = pd.concat([train_data_, selection])
                left_overs = np.array(list(set(train_data["id"]) - set(train_data_["id"])))
                left_overs = train_data[train_data["id"].isin(left_overs)].copy()
                if left_overs.shape[0] == 0:
                    break

            splitter = GroupShuffleSplit(n_splits=conf["trainer"]["n_splits"], 
                                         train_size=conf['trainer']['train_size2'], 
                                         random_state = seed)
            this_train_idx, this_valid_idx = list(splitter.split(train_data_, groups=train_data_['day']))[sidx]
            this_train_data, this_valid_data = train_data_.iloc[this_train_idx], train_data_.iloc[this_valid_idx]

            data_dict = {
                "train_data": this_train_data,
                "valid_data": this_valid_data,
                "test_data": test_data,
                "left_overs": left_overs
            }

            train_df, train_results, valid_results, test_results, left_overs = train(
                conf, data_dict, train_metric, direction
            )

            train_df.to_csv(
                os.path.join(save_loc, 
                             f"train_log_{iteration}_{sidx}.csv"),
                index = False
            )

            metrics = ["loss", "acc", "ave_acc", "ave_f1", "auc"]
            splits = ["train", "valid", "test"]
            result_dfs = [train_results, valid_results, test_results]

            print_str = f"Iteration {iteration}"
            active_results["iteration"].append(iteration)
            active_results["ensemble"].append(sidx)
            for metric in metrics:
                for _split, rdf in zip(splits, result_dfs):
                    value = np.mean(rdf[metric])
                    active_results[f"{_split}_{metric}"].append(value)
                    print_str += f" {_split}_{metric} {value:.4f}"

            active_df = pd.DataFrame.from_dict(active_results)
            active_df.to_csv(os.path.join(save_loc, f"active_train_log_{sidx}.csv"))
            print(print_str)
            #my_iter.set_description(print_str)
            #my_iter.refresh()
            
        if start_at_split > 0:
            break