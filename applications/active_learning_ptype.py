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

from evml.training import train_one_epoch, validate
#from torchmetrics import Accuracy

import random, os, numpy as np, sys, shutil, glob


def ave_acc(pred, true):
    conditions = [
        true == 0,
        true == 1,
        true == 2,
        true == 3
    ]

    score = np.mean([
        (true[condition] == pred[condition]).float().mean().cpu().item()
        for condition in conditions
    ])
    return score

def compute_metrics(results_dict, labels, outputs, preds):
    results_dict["acc"].append(torch.mean((preds == labels.data).float()).item())
    results_dict["ave_acc"].append(ave_acc(preds, labels.data))
    results_dict["ave_f1"].append(f1_score(labels.cpu(), preds.cpu(), average = "macro"))
    try:
        results_dict["auc"].append(roc_auc_score(labels.cpu(), 
                                                 outputs.detach().cpu(), 
                                                 multi_class='ovo', 
                                                 average = "macro"))
    except:
        pass
    return results_dict
    


def train_one_epoch(
    epoch,
    model,
    dataloader,
    num_classes,
    criterion,
    optimizer,
    batch_size,
    device=None,
    uncertainty=False,
    metric="accuracy",
    verbose=True
):

    if not device:
        device = get_device()

    if verbose:
        total = int(np.ceil(len(dataloader.dataset) / batch_size))
        my_iter = tqdm.tqdm(enumerate(dataloader),
                        total = total,
                        leave = True)
    else:
        my_iter = enumerate(dataloader)

    # Iterate over data.
    model.train()
    results_dict = defaultdict(list)
    
    for i, (inputs, labels) in my_iter:

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        if uncertainty:
            y = one_hot_embedding(labels, num_classes)
            y = y.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(
                outputs, y.float(), epoch, num_classes, 10, device
            )

            match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
            acc = torch.mean(match)
            evidence = relu_evidence(outputs)
            alpha = evidence + 1
            u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

            total_evidence = torch.sum(evidence, 1, keepdim=True)
            mean_evidence = torch.mean(total_evidence)
            mean_evidence_succ = torch.sum(
                torch.sum(evidence, 1, keepdim=True) * match
            ) / torch.sum(match + 1e-20)
            mean_evidence_fail = torch.sum(
                torch.sum(evidence, 1, keepdim=True) * (1 - match)
            ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

        else:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            prob = F.softmax(outputs, dim=1)

        # backprop
        loss.backward()
        optimizer.step()

        # statistics
        results_dict["loss"].append(loss.item())
        results_dict = compute_metrics(results_dict, labels, prob, preds)
        #results_dict["acc"].append(torch.mean((preds == labels.data).float()).item())
        #results_dict["ave_acc"].append(ave_acc(preds, labels.data))
        #results_dict["ave_f1"].append(f1_score(labels.cpu(), preds.cpu(), average = "macro"))
        #results_dict["auc"].append(roc_auc_score(labels.cpu(), outputs.detach().cpu(), multi_class='ovr', average = "macro"))
        
        if verbose:
            print_str = f"Epoch: {epoch} "
            print_str += f'train_loss: {np.mean(results_dict["loss"]):.4f} '
            print_str += f'train_acc: {np.mean(results_dict["acc"]):.4f}'
            print_str += f'train_ave_acc: {np.mean(results_dict["ave_acc"]):.4f}'
            my_iter.set_description(print_str)
            my_iter.refresh()

    return results_dict, model, optimizer


def validate(
    epoch,
    model,
    dataloader,
    num_classes,
    criterion,
    batch_size,
    device=None,
    uncertainty=False,
    metric="accuracy",
    verbose=True,
    return_preds=False
):

    if not device:
        device = get_device()

    if verbose:
        total = int(np.ceil(len(dataloader.dataset) / batch_size))
        my_iter = tqdm.tqdm(enumerate(dataloader),
                        total = total,
                        leave = True)
    else:
        my_iter = enumerate(dataloader)

    # Iterate over data.
    model.eval()
    results_dict = defaultdict(list)
    
    with torch.no_grad():
        for i, (inputs, labels) in my_iter:

            inputs = inputs.to(device)
            labels = labels.to(device)

            if uncertainty:
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(
                    outputs, y.float(), epoch, num_classes, 10, device
                )

                match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                acc = torch.mean(match)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1
                u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

                total_evidence = torch.sum(evidence, 1, keepdim=True)
                mean_evidence = torch.mean(total_evidence)
                mean_evidence_succ = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * match
                ) / torch.sum(match + 1e-20)
                mean_evidence_fail = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * (1 - match)
                ) / (torch.sum(torch.abs(1 - match)) + 1e-20)
                
                if return_preds:
                    results_dict["pred_uncertainty"].append(u)
                    results_dict["pred_labels"].append(preds.unsqueeze(-1))
                    results_dict["true_labels"].append(labels.unsqueeze(-1))
                    results_dict["pred_probs"].append(prob)

            else:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                prob = F.softmax(outputs, dim=1)
                
                if return_preds:
					### Add MC-dropout for "pred_uncertainty"
                    results_dict["pred_labels"].append(preds.unsqueeze(-1))
                    results_dict["true_labels"].append(labels.unsqueeze(-1))
                    results_dict["pred_probs"].append(prob)

            # statistics
            results_dict["loss"].append(loss.item())
            results_dict = compute_metrics(results_dict, labels, prob, preds)
            #results_dict["acc"].append(torch.mean((preds == labels.data).float()).item())
            #results_dict["ave_acc"].append(ave_acc(preds, labels.data))
            #results_dict["ave_f1"].append(f1_score(labels.data, preds.cpu(), average = "macro"))
            #results_dict["auc"].append(roc_auc_score(labels.cpu(), outputs.detach().cpu(), multi_class='ovr', average = "macro"))
            
            if verbose:
                print_str = f"Epoch: {epoch} "
                print_str += f'valid_loss: {np.mean(results_dict["loss"]):.4f} '
                print_str += f'valid_acc: {np.mean(results_dict["acc"]):.4f}'
                my_iter.set_description(print_str)
                my_iter.refresh()
                
    if return_preds:
        if "pred_uncertainty" in results_dict:
            results_dict["pred_uncertainty"] = torch.cat(results_dict["pred_uncertainty"], 0)
        results_dict["pred_probs"] = torch.cat(results_dict["pred_probs"], 0)
        results_dict["pred_labels"] = torch.cat(results_dict["pred_labels"], 0)
        results_dict["true_labels"] = torch.cat(results_dict["true_labels"], 0)

    return results_dict


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def train(conf, data, train_metric = "valid_ave_f1", direction = "max"):
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
        weights = torch.from_numpy(np.array(outputvar_weights)).float().to(device)
        weights /= weights.sum()
        criterion = nn.CrossEntropyLoss(
            weight = weights,
            #label_smoothing = label_smoothing
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
        train_metric = "valid_auc"
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
    #model = DNN(
    #    len(features), 
    #    len(outputs), 
    #    block_sizes = [middle_size for _ in range(num_hidden_layers)], 
    #    dr = [dropout_rate for _ in range(num_hidden_layers)], 
    #    batch_norm = batch_norm, 
    #    lng = False
    #).to(device)
	#model.load_weights(f"{save_loc}/best.pt")
        
    #checkpoint = torch.load(
    #    f"{save_loc}/best.pt",
    #    map_location=lambda storage, loc: storage
    #)
	
    #epoch = checkpoint["epoch"]
    
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
    if use_uncertainty:
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
                elif policy == "uncertainty":
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

            train_df, train_results, valid_results, test_results, left_overs = train(conf, data_dict)

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