import numpy as np
import tqdm
from collections import defaultdict
import torch
import torch.nn.functional as F
from evml.class_losses import *
from evml.metrics import compute_metrics


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def train_one_epoch(
    epoch,
    model,
    dataloader,
    num_classes,
    criterion,
    optimizer,
    batch_size,
    weights=None,
    device=None,
    uncertainty=False,
    metric="accuracy",
    verbose=True,
):

    if not device:
        device = get_device()

    if verbose:
        total = int(np.ceil(len(dataloader.dataset) / batch_size))
        my_iter = tqdm.tqdm(enumerate(dataloader), total=total, leave=True)
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
            loss = criterion(outputs, y.float(), epoch, num_classes, 10, weights, device)

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
        # results_dict["acc"].append(torch.mean((preds == labels.data).float()).item())
        # results_dict["ave_acc"].append(ave_acc(preds, labels.data))
        # results_dict["ave_f1"].append(f1_score(labels.cpu(), preds.cpu(), average = "macro"))
        # results_dict["auc"].append(roc_auc_score(labels.cpu(), outputs.detach().cpu(), multi_class='ovr', average = "macro"))

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
    weights=None,
    device=None,
    uncertainty=False,
    metric="accuracy",
    verbose=True,
    return_preds=False,
):

    if not device:
        device = get_device()

    if verbose:
        total = int(np.ceil(len(dataloader.dataset) / batch_size))
        my_iter = tqdm.tqdm(enumerate(dataloader), total=total, leave=True)
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
                loss = criterion(outputs, y.float(), epoch, num_classes, 10, weights, device)

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
            # results_dict["acc"].append(torch.mean((preds == labels.data).float()).item())
            # results_dict["ave_acc"].append(ave_acc(preds, labels.data))
            # results_dict["ave_f1"].append(f1_score(labels.data, preds.cpu(), average = "macro"))
            # results_dict["auc"].append(roc_auc_score(labels.cpu(), outputs.detach().cpu(), multi_class='ovr', average = "macro"))

            if verbose:
                print_str = f"Epoch: {epoch} "
                print_str += f'valid_loss: {np.mean(results_dict["loss"]):.4f} '
                print_str += f'valid_acc: {np.mean(results_dict["acc"]):.4f}'
                my_iter.set_description(print_str)
                my_iter.refresh()

    if return_preds:
        if "pred_uncertainty" in results_dict:
            results_dict["pred_uncertainty"] = torch.cat(
                results_dict["pred_uncertainty"], 0
            )
        results_dict["pred_probs"] = torch.cat(results_dict["pred_probs"], 0)
        results_dict["pred_labels"] = torch.cat(results_dict["pred_labels"], 0)
        results_dict["true_labels"] = torch.cat(results_dict["true_labels"], 0)

    return results_dict
