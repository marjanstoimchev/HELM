import numpy as np
import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from data.chestxray8 import ChestXray8ImageReader, ChestXray8
from data.nihchestxray import NIHChestXrayImageReader, NIHChestXray
from data.hpa import HPAImageReader, HPA
from data.padchest import PadChestImageReader, PadChest
from data.mured import MuRedImageReader, MuRed
from data.huggingface import HFImageReader, HuggingFace
from data.dfc_15 import DFC_15ImageReader, DFC_15

# Dataset Factory to choose appropriate dataset class
class DatasetFactory:
    @staticmethod
    def create_dataset(dataset_name, num_classes):
        if dataset_name == 'dfc_15':
            return DFC_15()
        elif dataset_name == 'MuRed':
            return MuRed()
        elif dataset_name == 'ChestX-ray8':
            return ChestXray8()
        elif dataset_name == 'PadChest':
            return PadChest()
        elif dataset_name == 'HPA':
            return HPA()
        elif dataset_name == 'NIHChestXray':
            return NIHChestXray()
        elif dataset_name in ['AID_Multilabel', 'UC_Merced_LandUse_Multilabel', 'MLRSNet']:
            return HuggingFace(dataset_name, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

def get_image_reader(dataset_name):
    if dataset_name == 'ChestX-ray8':
        return ChestXray8ImageReader()
    elif dataset_name == 'dfc_15':
        return DFC_15ImageReader()
    elif dataset_name == 'NIHChestXray':
        return NIHChestXrayImageReader()
    elif dataset_name == 'HPA':
        return HPAImageReader()
    elif dataset_name == 'PadChest':
        return PadChestImageReader()
    elif dataset_name == 'MuRed':
        return MuRedImageReader()
    elif dataset_name in ['AID_Multilabel', 'UC_Merced_LandUse_Multilabel', 'MLRSNet']:
            return HFImageReader()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
# Function to extend columns based on label names
def extend_columns(df, label_names):
    # Create a dictionary to map the index to label names
    label_map = {i: label for i, label in enumerate(label_names)}
    
    # Add new columns to the dataframe and initialize them with 0
    for key in label_map.keys():
        df[label_map[key]] = 0
    
    # Pass the label map to the 'fill_targets' function via a lambda function
    df = df.apply(lambda row: fill_targets(row, label_map), axis=1)
    return df

# Function to fill in the target columns based on the label array
def fill_targets(row, label_map):    
    target_array = np.array(row.label).astype(np.int64)
    # Set the corresponding columns to 1 for the present labels
    for num in target_array:
        if int(num) in label_map:
            name = label_map[int(num)]
            row[name] = 1
    return row

# Utility Functions for dataset splitting

def random_sampling(df: pd.DataFrame, p: float, seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly splits a DataFrame into labeled and unlabeled sets.
    """
    if seed is not None:
        np.random.seed(seed)

    n_labeled = int(np.floor(len(df) * p))
    if p >= 1.0:
        n_labeled = len(df)
    else:
        n_labeled = max(min(n_labeled, len(df) - 1), 1) if len(df) > 1 else len(df)

    indices = np.random.permutation(df.index)
    indices_labeled = indices[:n_labeled]
    indices_unlabeled = indices[n_labeled:]
    
    return indices_labeled, indices_unlabeled

def split_dataset_(df, test_size=0.2, seed=42):
    """
    Performs a stratified split on multilabel data using MultilabelStratifiedShuffleSplit.

    Parameters:
    - df (pd.DataFrame): DataFrame containing multilabel data.
    - test_size (float): Proportion of the dataset to use as the test set.
    - seed (int): Random seed for reproducibility.

    Returns:
    - tuple: Indices for train and test splits.
    """
    df = df.reset_index(drop=True)
    X, y = df.image, np.stack(df.one_hot)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(msss.split(X, y))
    return train_idx, test_idx

from sklearn.model_selection import GroupShuffleSplit 

def split_dataset_medical(df, test_size=0.2, seed=42):
    """
    Performs a stratified split on multilabel data using GroupShuffleSplit.
    """

    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state = seed)
    split = splitter.split(df, groups=df['Patient ID'])
    train_idx, test_idx = next(split)
    return train_idx, test_idx

def prepare_train_test_validation(df, test_size=0.2, val_size=0.1, seed=42):
    """
    Prepares train, test, and optionally validation splits from a DataFrame. If splits are not predefined,
    performs stratified splitting using MultilabelStratifiedShuffleSplit.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'split' column with predefined splits or multilabel data for stratified splitting.
    - test_size (float): Proportion of the dataset to use as the test set (if split not predefined).
    - val_size (float): Proportion of the entire dataset to use as the validation set.
    - seed (int): Random seed for reproducibility.

    Returns:
    - dict: Dictionary containing train, test, and optionally validation indices.
    """
    splits = {}

    # Use predefined test split if available
    if 'split' in df.columns and 'test' in df['split'].unique():
        splits['test'] = list(df.index[df['split'] == 'test'])
        train_idx = list(df.index[df['split'] != 'test'])  # Assume all non-test data is trainable
    else:
        # Perform stratified split for test if not predefined
        train_idx, test_idx = split_dataset_(df, test_size=test_size, seed=seed)
        splits['test'] = test_idx

    # Use predefined validation split if available
    if 'split' in df.columns and 'val' in df['split'].unique():
        splits['val'] = list(df.index[df['split'] == 'val'])
    else:
        if val_size > 0:
            # Further split train into train and validation
            train_df = df.iloc[train_idx]
            train_idx, val_idx = split_dataset_(train_df, test_size=val_size, seed=seed)
            splits['val'] = val_idx

    # Use predefined train split if available, otherwise use the remaining data
    if 'split' in df.columns and 'train' in df['split'].unique():
        splits['train'] = list(df.index[df['split'] == 'train'])
    else:
        splits['train'] = train_idx

    return splits





