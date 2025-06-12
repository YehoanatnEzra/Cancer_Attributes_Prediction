# test_linear_regression.py
from src.load_and_split import load_and_split_part2
from src.process_data import remove_cols, clean_data
from src.models import get_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os

output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

train_feats = "../../data/train_test_splits/train.feats.csv"
labels_1 = "../../data/train_test_splits/train.labels.1.csv"
test_path = "../../data/train_test_splits/test.feats.csv"
model_name = "elasticnet"
task_type = "regression"

print(f"\nTesting {model_name.title()} Regressor (Part 2)...")

for seed in [0, 1, 2, 3, 4]:
    print(f"\nSeed: {seed}")
    x_train, x_dev, y_train_raw, y_dev_raw = load_and_split_part2(train_feats, labels_1, seed)

    x_train = clean_data(remove_cols(x_train))
    x_dev = clean_data(remove_cols(x_dev))

    y_train = y_train_raw.iloc[:, 0].astype(float).values
    y_dev = y_dev_raw.iloc[:, 0].astype(float).values

    model = get_model(task_type, model_name)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_dev)

    mse = mean_squared_error(y_dev, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    df_dev = pd.DataFrame({"tumor_size": y_pred})
    df_dev.to_csv(os.path.join(output_dir, f"predictions_{model_name}_seed{seed}.csv"), index=False)

    x_test = pd.read_csv(test_path)
    x_test_cleaned = clean_data(remove_cols(x_test))
    y_test_pred = model.predict(x_test_cleaned)

    df_test = pd.DataFrame({"tumor_size": y_test_pred})
    df_test.to_csv(os.path.join(output_dir, f"test_predictions_{model_name}_seed{seed}.csv"), index=False)
    print(f"Saved test predictions to outputs/test_predictions_{model_name}_seed{seed}.csv")
