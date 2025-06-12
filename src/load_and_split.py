"""
Module for loading and splitting data for the IML Hackathon project.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.labels_utils import LabelEncoder


def flatten(lst):
    """
    Flatten a nested list
    """
    return [item for sublist in lst for item in sublist]

def initilaize_labels(Y):
    """
    Initialize labels and create mapping dictionaries
    """
    global label_to_ind
    global ind_to_label
    global num_of_label
    col_name = Y.columns[0]
    ls_origin_labels = [eval(val) for val in Y[col_name]]
    labs = list(set(flatten(ls_origin_labels)))
    inds = list(range(len(labs)))
    label_to_ind = dict(zip(labs, inds))
    ind_to_label = dict(zip(inds, labs))
    num_of_label = len(labs)

def load_and_split_part1(filename_train, filename_labels_0, seed=42):
    """
    Load and split the data for part 1 of the task

    Args:
        filename_train (str): Path to the training data CSV file
        filename_labels_0 (str): Path to the labels_0 CSV file

    Returns:
        tuple: (x_train, x_dev, labels_0_train, labels_0_dev)
    """
    # Load training data
    train = pd.read_csv(filename_train)
    ind = train.duplicated(subset=['id-hushed_internalpatientid', 'אבחנה-Diagnosis date'])
    train = train[~ind]

    # Load and process labels_0
    train_labels_0 = pd.read_csv(filename_labels_0)
    train_labels_0 = train_labels_0[~ind]

    # Use LabelEncoder
    encoder = LabelEncoder()
    encoder.initialize(train_labels_0)

    # Split the data
    x_train, x_dev, labels_0_train, labels_0_dev = train_test_split(
        train, train_labels_0,
        test_size=0.25,
        random_state=seed
    )

    return x_train, x_dev, labels_0_train, labels_0_dev, encoder

def load_and_split_part2(filename_train, filename_labels_1, seed=42):
    """
    Load and split the data for part 2 of the task
    
    Args:
        filename_train (str): Path to the training data CSV file
        filename_labels_1 (str): Path to the labels_1 CSV file
        
    Returns:
        tuple: (x_train, x_dev, labels_1_train, labels_1_dev)
    """
    # Load training data
    train = pd.read_csv(filename_train)
    ind = train.duplicated(subset=['id-hushed_internalpatientid', 'אבחנה-Diagnosis date'])
    train = train[~ind]

    # Load and process labels_1
    train_labels_1 = pd.read_csv(filename_labels_1)
    train_labels_1 = train_labels_1[~ind]

    # Split the data
    x_train, x_dev, labels_1_train, labels_1_dev = train_test_split(
        train, train_labels_1,
        test_size=0.25,
        random_state=seed
    )

    return x_train, x_dev, labels_1_train, labels_1_dev