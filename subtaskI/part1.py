import sys
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from load_and_split import split_part1_data
from sklearn.ensemble import RandomForestClassifier
from process_data import remove_cols, clean_data
from sklearn.metrics import f1_score



def part_1(train_path, labels_0_path, test_path, output_dir):
    print("\n[Part 1] Multi-label Classification")
    # Load the gold file temporarily to get the column name
    gold_df = pd.read_csv(labels_0_path)
    col_name = gold_df.columns[0]

    x_train, x_dev, y_train_raw, y_dev_raw, encoder = split_part1_data(train_path, labels_0_path, seed=3)
    x_train = clean_data(remove_cols(x_train))
    x_dev = clean_data(remove_cols(x_dev))
    y_train = np.array([encoder.to_binary_vector(eval(row)) for row in y_train_raw.iloc[:, 0]])
    y_dev = np.array([encoder.to_binary_vector(eval(row)) for row in y_dev_raw.iloc[:, 0]])

    clf = RandomForestClassifier(random_state=1, n_estimators=200)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_dev)

    print("F1 Micro:", f1_score(y_dev, y_pred, average="micro", zero_division=0))
    print("F1 Macro:", f1_score(y_dev, y_pred, average="macro", zero_division=0))

    # Predict on test
    x_test = pd.read_csv(test_path, low_memory=False, dtype=str)
    x_test = clean_data(remove_cols(x_test))
    y_test_pred = clf.predict(x_test)
    y_test_pred_str = [encoder.from_binary_vector(row) for row in y_test_pred]
    df_out = pd.DataFrame({col_name: [str(labs) for labs in y_test_pred_str]})
    df_out.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python part1.py <train_feats> <labels> <test_feats>")
        sys.exit(1)
    print(sys.argv)
    train_path = sys.argv[1]
    labels_0_path = sys.argv[2]
    test_path = sys.argv[3]

    np.random.seed(0)

    part_1(train_path, labels_0_path, test_path, "./subtaskI")


