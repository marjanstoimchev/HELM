import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_score,
    f1_score, 
    recall_score, 
    roc_auc_score,
    average_precision_score,
    label_ranking_loss,
    coverage_error,
    accuracy_score,
    hamming_loss
    )

class Dotdict:
    """Wraps dictionaries to allow value access in dot notation.

    Instead of data[key], access value as data.key"""

    def __init__(self, data: dict):
        super().__init__()
        for k, v in data.items():
            if isinstance(v, dict):
                # take care of nested dicts
                v = Dotdict(v)
            self.__dict__[k] = v

    def to_dict(self):
        """Converts Dotdict to a regular dictionary."""
        return {k: (v.to_dict() if isinstance(v, Dotdict) else v) for k, v in self.__dict__.items()}

def one_hot_to_name(df: pd.DataFrame, hierarchy_dict: dict, id: int) -> dict:
    """
    Convert one-hot encoded values to their corresponding label names and strings.

    Parameters:
    df (pd.DataFrame): DataFrame containing one-hot encoded vectors in the 'h_one_hot' column.
    hierarchy_dict (dict): Dictionary mapping label names to their corresponding strings.
    id (int): Row identifier to locate the one-hot encoded vector.

    Returns:
    dict: A dictionary mapping label names to corresponding strings based on the one-hot vector.
    """

    # Retrieve one-hot encoded vector for the given id
    if id not in df.index:
        raise ValueError(f"ID {id} not found in DataFrame.")

    one_hot_vector = df.loc[id, 'h_one_hot']

    # Ensure the one-hot vector exists and is iterable
    if not isinstance(one_hot_vector, (list, np.ndarray)):
        raise ValueError(f"One-hot vector at ID {id} is not in a valid format.")

    # Find indices where the one-hot vector has a value of 1
    indices = np.where(np.array(one_hot_vector) == 1)[0]

    # Extract label names and corresponding strings based on the one-hot indices
    label_names = list(hierarchy_dict.keys())
    label_strings = list(hierarchy_dict.values())

    if len(label_names) != len(label_strings):
        raise ValueError("Hierarchy dictionary keys and values lengths do not match.")

    # Map the selected labels and strings into a dictionary
    return {label_names[i]: label_strings[i] for i in indices}


def one_hot_np_to_name(y, hierarchy_dict: dict) -> dict:
    """
    Convert one-hot encoded values to their corresponding label names and strings.

    Parameters:
    y (np.array): array containing one-hot encoded vectors.
    hierarchy_dict (dict): Dictionary mapping label names to their corresponding strings.
    id (int): Row identifier to locate the one-hot encoded vector.

    Returns:
    dict: A dictionary mapping label names to corresponding strings based on the one-hot vector.
    """

    one_hot_vector = y

    # Find indices where the one-hot vector has a value of 1
    indices = np.where(np.array(one_hot_vector) == 1)[0]

    # Extract label names and corresponding strings based on the one-hot indices
    label_names = list(hierarchy_dict.keys())
    label_strings = list(hierarchy_dict.values())

    if len(label_names) != len(label_strings):
        raise ValueError("Hierarchy dictionary keys and values lengths do not match.")

    # Map the selected labels and strings into a dictionary
    return {label_names[i]: label_strings[i] for i in indices}

def findmax(outputs):
    Max = -float("inf")
    index = 0
    for i in range(outputs.shape[0]):
        if outputs[i] > Max:
            Max = outputs[i]
            index = i
    return Max, index

def OneError(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    num = 0
    one_error = 0
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            Max, index = findmax(outputs[i])
            num = num + 1
            if test_target[i][index] != 1:
                one_error = one_error + 1
    return one_error / num

def calculate_metrics(Y):
    y_true, y_pred, y_scores = Y['y_true'], Y['y_pred'], Y['y_scores']
                
    metric_names = ["ranking loss", "one error", "coverage",
                    "average auprc", "weighted auprc", #"average auroc",
                    "micro f1", "micro recall", "micro precision",
                    "macro f1", "macro recall", "macro precision",
                    "subset accuracy", "hamming loss",
                    "ml_f_one", "ml_recall", "ml_precision"
                    ]
        
    r_loss = label_ranking_loss(y_true, y_scores)
    oe = OneError(y_scores, y_true)
    coverage = coverage_error(y_true, y_scores)
    
    average_auprc = average_precision_score(y_true, y_scores, average="macro")
    weighted_auprc = average_precision_score(y_true, y_scores, average="weighted")
    # average_auroc = roc_auc_score(y_true, y_scores, average="macro") 
    
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average = "micro")
    micro_precision = precision_score(y_true, y_pred, average = "micro")
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average = "macro")
    macro_precision = precision_score(y_true, y_pred, average = "macro", zero_division=0)
    
    subset_accuracy = accuracy_score(y_true, y_pred) # exact match
    hl = hamming_loss(y_true, y_pred)
    
    ml_f_one     = f1_score(y_true, y_pred, average='samples')
    ml_recall    = recall_score(y_true, y_pred, average = "samples")
    ml_precision = precision_score(y_true, y_pred, average = "samples", zero_division=0)

    metrics = [r_loss, oe, coverage,
               average_auprc, weighted_auprc, #average_auroc,
               micro_f1, micro_recall, micro_precision,
               macro_f1, macro_recall, macro_precision,
               subset_accuracy, hl,
               ml_f_one, ml_recall, ml_precision]
    
    dict_metrics = {name: [metric] for name, metric in zip(metric_names, metrics)}
    df = pd.DataFrame(dict_metrics).T        
    return df

def predict(trainer, lightning_model, datamodule):
    Y = dict()
    test_outputs = trainer.predict(lightning_model, datamodule)
    logits = torch.cat([x['logits'] for x in test_outputs])
    labels = torch.cat([x['labels'] for x in test_outputs]).to(torch.int).numpy()
    scores = torch.sigmoid(logits).numpy()    
    preds = (scores > 0.5).astype('int')
    Y['y_scores'] = scores
    Y['y_pred']   = preds
    Y['y_true']   = labels
    return Y


