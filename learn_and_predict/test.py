# test.py
from src.load_and_split import load_and_split_part1, load_and_split_part2
from src.process_data import remove_cols, clean_data
from src.models import get_model
from src.labels_utils import LabelEncoder

from sklearn.metrics import f1_score, mean_squared_error
import pandas as pd
import numpy as np

# File paths
train_feats = "../data/train_test_splits/train.feats.csv"
labels_0 = "../data/train_test_splits/train.labels.0.csv"
labels_1 = "../data/train_test_splits/train.labels.1.csv"

### ---- PART 1: CLASSIFICATION (Multi-label Metastases) ---- ###
print("\n--- Part 1: Classification ---")

# Load + encode
x_train, x_dev, y_train_raw, y_dev_raw, encoder = load_and_split_part1(train_feats, labels_0)

# Clean
x_train = clean_data(remove_cols(x_train))
x_dev = clean_data(remove_cols(x_dev))

# Convert labels to binary multi-hot
y_train = np.array([encoder.to_binary_vector(eval(l)) for l in y_train_raw.iloc[:, 0]])
y_dev = np.array([encoder.to_binary_vector(eval(l)) for l in y_dev_raw.iloc[:, 0]])

# Train model
clf = get_model("classification", "rf")
clf.fit(x_train, y_train)

# Predict
y_pred = clf.predict(x_dev)

# Evaluate
f1_micro = f1_score(y_dev, y_pred, average="micro")
f1_macro = f1_score(y_dev, y_pred, average="macro")
print(f"Micro F1: {f1_micro:.4f}")
print(f"Macro F1: {f1_macro:.4f}")

### ---- PART 2: REGRESSION (Tumor Size) ---- ###
print("\n--- Part 2: Regression ---")

# Load
x_train, x_dev, y_train, y_dev = load_and_split_part2(train_feats, labels_1)

# Clean
x_train = clean_data(remove_cols(x_train))
x_dev = clean_data(remove_cols(x_dev))

# Flatten label arrays
y_train = y_train.iloc[:, 0].astype(float).values
y_dev = y_dev.iloc[:, 0].astype(float).values

# Train model
reg = get_model("regression", "boost")
reg.fit(x_train, y_train)

# Predict
y_pred = reg.predict(x_dev)

# Evaluate
mse = mean_squared_error(y_dev, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
