import pandas as pd
from sklearn.model_selection import train_test_split
from labels_utils import LabelEncoder


def flatten_nested_list(nested_list):
    """
    Unroll a nested list into a flat list
    """
    return [element for group in nested_list for element in group]


def setup_label_mappings(label_df):
    """
    Create global label-to-index and index-to-label mappings
    """
    global label_to_ind, ind_to_label, num_of_label
    main_col = label_df.columns[0]
    parsed_labels = [eval(x) for x in label_df[main_col]]
    unique_labels = sorted(set(flatten_nested_list(parsed_labels)))
    indices = list(range(len(unique_labels)))
    label_to_ind = dict(zip(unique_labels, indices))
    ind_to_label = dict(zip(indices, unique_labels))
    num_of_label = len(unique_labels)


def split_part1_data(train_file, labels_file_0, seed=42):
    """
    Load, clean, and split data for metastases prediction (Part 1)

    Args:
        train_file (str): Path to the train features CSV
        labels_file_0 (str): Path to multi-label target CSV

    Returns:
        tuple: train/dev features, train/dev labels, label encoder
    """
    data = pd.read_csv(train_file, dtype=str, low_memory=False)
    duplicate_mask = data.duplicated(subset=['id-hushed_internalpatientid', 'אבחנה-Diagnosis date'])
    data = data[~duplicate_mask]

    targets = pd.read_csv(labels_file_0, dtype=str, low_memory=False)
    targets = targets[~duplicate_mask]

    encoder = LabelEncoder()
    encoder.initialize(targets)

    x_train, x_val, y_train, y_val = train_test_split(
        data, targets, test_size=0.25, random_state=3
    )

    return x_train, x_val, y_train, y_val, encoder


def split_part2_data(train_file, labels_file_1, seed=42):
    """
    Load and prepare data for tumor size regression (Part 2)

    Args:
        train_file (str): Path to the feature CSV
        labels_file_1 (str): Path to the regression label CSV

    Returns:
        tuple: split features and labels
    """
    data = pd.read_csv(train_file, dtype=str,low_memory=False)
    dup_mask = data.duplicated(subset=['id-hushed_internalpatientid', 'אבחנה-Diagnosis date'])
    data = data[~dup_mask]

    targets = pd.read_csv(labels_file_1, dtype=str, low_memory=False)
    targets = targets[~dup_mask]

    x_train, x_val, y_train, y_val = train_test_split(
        data, targets, test_size=0.25, random_state=3
    )

    return x_train, x_val, y_train, y_val
