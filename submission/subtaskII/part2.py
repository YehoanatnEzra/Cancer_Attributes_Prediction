import sys
import os
import numpy as np
import pandas as pd
from load_and_split import split_part2_data
from sklearn.linear_model import LinearRegression
from process_data import remove_cols, clean_data
from sklearn.metrics import mean_squared_error



def part_2(train_path, labels_1_path, test_path, output_dir):
    print("\n[Part 2] Tumor Size Regression")
    # Load the gold file temporarily to get the column name
    gold_df = pd.read_csv(labels_1_path)
    col_name = gold_df.columns[0]

    x_train, x_dev, y_train, y_dev = split_part2_data(train_path, labels_1_path, seed=0)
    x_train = clean_data(remove_cols(x_train))
    x_dev = clean_data(remove_cols(x_dev))
    model = LinearRegression(tol=1e-6)
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_dev)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    print("MSE:", mean_squared_error(y_dev, y_pred))

    x_test = pd.read_csv(test_path, low_memory=False, dtype=str)
    x_test = clean_data(remove_cols(x_test))
    y_test_pred = model.predict(x_test)
    y_test_pred = np.clip(y_test_pred, a_min=0, a_max=None)
    df_out = pd.DataFrame({col_name: [str(labs) for labs in y_test_pred]})
    df_out.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)




if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python part2.py <train_feats> <labels> <test_feats>")
        sys.exit(1)

    train_path = sys.argv[1]
    labels_1_path = sys.argv[2]
    test_path = sys.argv[3]

    np.random.seed(0)

    part_2(train_path, labels_1_path, test_path, "./subtaskII")

