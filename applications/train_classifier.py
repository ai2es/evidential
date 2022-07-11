import logging, tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader

import copy
import time, yaml
import torch.nn.functional as F
import pandas as pd, numpy as np
pd.options.mode.chained_assignment = None

from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

from evml.reliability import reliability_diagram, reliability_diagrams, compute_calibration
from evml.class_losses import *


import random, os, numpy as np


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


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

        # backprop
        loss.backward()
        optimizer.step()

        # statistics
        results_dict["loss"].append(loss.item())
        results_dict["acc"].append(torch.mean((preds == labels.data).float()).item())
        
        if verbose:
            print_str = f"Epoch: {epoch} "
            print_str += f'train_loss: {np.mean(results_dict["loss"]):.4f} '
            print_str += f'train_acc: {np.mean(results_dict["acc"]):.4f}'
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
                prob = F.softmax(output, dim=1)
                
                if return_preds:
                    results_dict["pred_labels"].append(preds.unsqueeze(-1))
                    results_dict["true_labels"].append(labels.unsqueeze(-1))
                    results_dict["pred_probs"].append(prob)

            # statistics
            results_dict["loss"].append(loss.item())
            results_dict["acc"].append(torch.mean((preds == labels.data).float()).item())
            
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


def load_mlp_model(input_size, middle_size, output_size, dropout):
    model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_size, middle_size)),
            #nn.BatchNorm1d(middle_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(middle_size, output_size))
    ) 
    return model


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


if __name__ == "__main__":


    config = "config/p-type.yml"
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    features = conf['tempvars'] + conf['tempdewvars'] + conf['ugrdvars'] + conf['vgrdvars']
    outputs = conf['outputvars']
    num_classes = len(outputs)
    n_splits = conf['trainer']['n_splits']
    train_size1 = conf['trainer']['train_size1'] # sets test size
    train_size2 = conf['trainer']['train_size2'] # sets valid size
    num_hidden_layers = conf['trainer']['num_hidden_layers']
    hidden_sizes = conf['trainer']['hidden_sizes']
    dropout_rate = conf['trainer']['dropout_rate']
    batch_size = conf['trainer']['batch_size']
    learning_rate = conf['trainer']['learning_rate']
    metrics = conf['trainer']['metrics']
    run_eagerly = conf['trainer']['run_eagerly']
    shuffle = conf['trainer']['shuffle']
    epochs = conf['trainer']['epochs']

    lr_patience = 3
    stopping_patience = 10
    verbose = True
    save_loc = "results/categorical"
    middle_size = 100
    learning_rate = 1e-3
    L2_reg = 0.005
    seed = 1000

    loss = "digamma"
    use_uncertainty = False if loss == "ce" else True
    
    ### Set the seed for reproducibility
    seed_everything(seed)

    ### Load the data
    df = pd.read_parquet(conf['data_path'])

    ### Split and preprocess the data
    df['day'] = df['datetime'].apply(lambda x: str(x).split(' ')[0])

    splitter = GroupShuffleSplit(n_splits=n_splits, train_size=train_size1, random_state = seed)
    train_idx, test_idx = list(splitter.split(df, groups=df['day']))[0]
    train_data, test_data = df.iloc[train_idx], df.iloc[test_idx]

    splitter = GroupShuffleSplit(n_splits=n_splits, train_size=train_size2)
    train_idx, valid_idx = list(splitter.split(train_data, groups=train_data['day']))[0]
    train_data, valid_data = train_data.iloc[train_idx], train_data.iloc[valid_idx]

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(train_data[features])
    x_valid = scaler_x.transform(valid_data[features])
    x_test = scaler_x.transform(test_data[features])
    y_train = np.argmax(train_data[outputs].to_numpy(), 1)
    y_valid = np.argmax(valid_data[outputs].to_numpy(), 1)
    y_test = np.argmax(test_data[outputs].to_numpy(), 1)

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
        criterion = nn.CrossEntropyLoss()

    ### Load MLP model
    model = load_mlp_model(len(features), middle_size, len(outputs), dropout_rate)
    model = model.to(device)
    
    ### Initialize an optimizer
    optimizer = optim.Adam(model.parameters(), 
                           lr=learning_rate, 
                           weight_decay=L2_reg)

    ### Load a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience = lr_patience, 
        verbose = verbose,
        min_lr = 1.0e-13
    )
    
    ### Train the model
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
            return_preds=False
        )
        
        results_dict["train_loss"].append(np.mean(train_results["loss"]))
        results_dict["train_acc"].append(np.mean(valid_results["acc"]))
        results_dict["valid_loss"].append(np.mean(train_results["loss"]))
        results_dict["valid_acc"].append(np.mean(valid_results["acc"]))
        
        ### Save the dataframe to disk
        df = pd.DataFrame.from_dict(results_dict).reset_index()
        df.to_csv(f"{save_loc}/training_log.csv", index = False)
        
        ### Save the model only if its the best
        if results_dict["valid_acc"][-1] == max(results_dict["valid_acc"]):
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_x': scaler_x,
                'valid_accuracy': min(results_dict["valid_acc"])
            }
            torch.save(state_dict, f"{save_loc}/best.pt")
        
        ### Update the scheduler
        lr_scheduler.step(1.0 - results_dict["valid_acc"][-1])
        
        ### Early stopping
        best_epoch = [i for i,j in enumerate(
            results_dict["valid_acc"]) if j == max(results_dict["valid_acc"])][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break
            
            
    ### Evaluate with the best model
    model = load_mlp_model(len(features), middle_size, len(outputs), dropout_rate).to(device)
        
    checkpoint = torch.load(
        f"{save_loc}/best.pt",
        map_location=lambda storage, loc: storage
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    model.eval()
    
    ### Wrap the test split
    test_split = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_split, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=0)
    
    test_results = validate(
            epoch,
            model,
            test_loader,
            num_classes,
            criterion,
            batch_size,
            device=device,
            uncertainty=use_uncertainty,
            return_preds=True
        )
    
    ### Add the predictions to the original dataframe and save to disk
    for idx in range(len(outputs)):
        test_data[f"{outputs[idx]}_conf"] = test_results["pred_probs"][:, idx].cpu().numpy()
    test_data["uncertainty"] = test_results["pred_uncertainty"][:, 0].cpu().numpy()
    test_data["pred_labels"] = test_results["pred_labels"][:, 0].cpu().numpy()
    test_data["true_labels"] = test_results["true_labels"][:, 0].cpu().numpy()
    test_data["pred_conf"] = np.max(test_results["pred_probs"].cpu().numpy(), 1)
    test_data.to_parquet(f"{save_loc}/test_data.parquet")