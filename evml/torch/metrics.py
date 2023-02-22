import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def ave_acc(pred, true):
    conditions = [true == 0, true == 1, true == 2, true == 3]

    score = np.mean(
        [
            (true[condition] == pred[condition]).float().mean().cpu().item()
            for condition in conditions
        ]
    )
    return score


def compute_metrics(results_dict, labels, outputs, preds):
    results_dict["acc"].append(torch.mean((preds == labels.data).float()).item())
    results_dict["ave_acc"].append(ave_acc(preds, labels.data))
    results_dict["ave_f1"].append(f1_score(labels.cpu(), preds.cpu(), average="macro"))
    try:
        results_dict["auc"].append(
            roc_auc_score(
                labels.cpu(), outputs.detach().cpu(), multi_class="ovo", average="macro"
            )
        )
    except Exception as e:
        print(e)
        pass
    return results_dict
