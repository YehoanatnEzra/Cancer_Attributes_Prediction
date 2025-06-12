import pandas as pd
from sklearn.model_selection import train_test_split

def initilaize_labels(labels_df):
    """
    Initialize labels dataframe by setting all values to 0
    """
    for col in labels_df.columns:
        if col != 'id-hushed_internalpatientid':
            labels_df[col] = 0

def load_and_split_part1(filename_train, filename_labels_0):
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
    initilaize_labels(train_labels_0)
    train_labels_0 = train_labels_0[~ind]

    # Split the data
    x_train, x_dev, labels_0_train, labels_0_dev = train_test_split(
        train, train_labels_0,
        test_size=0.25,
        random_state=42
    )

    return x_train, x_dev, labels_0_train, labels_0_dev

def load_and_split_part2(filename_train, filename_labels_1):
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
        random_state=42
    )

    return x_train, x_dev, labels_1_train, labels_1_dev 