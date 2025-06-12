from load_and_split import split_part1_data
from process_data import remove_cols, clean_data
from src.models import get_model
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import os

# Ensure outputs directory exists
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

train_feats = "../../data/train_test_splits/train.feats.csv"
labels_0 = "../../data/train_test_splits/train.labels.0.csv"

model_name = "rf"

print("\n Testing Random Forest (multi-label classification)...")

for seed in [0]:
    print(f"\nSeed: {seed}")
    x_train, x_dev, y_train_raw, y_dev_raw, encoder = split_part1_data(train_feats, labels_0, seed)

    # immediately after the split
    x_train, x_dev = x_train.reset_index(drop=True), x_dev.reset_index(drop=True)
    y_train_raw, y_dev_raw = y_train_raw.reset_index(drop=True), y_dev_raw.reset_index(drop=True)

    # Clean features
    x_train = clean_data(remove_cols(x_train))
    x_dev = clean_data(remove_cols(x_dev))

    # Convert labels
    y_train = np.array([encoder.to_binary_vector(eval(row)) for row in y_train_raw.iloc[:, 0]])
    y_dev = np.array([encoder.to_binary_vector(eval(row)) for row in y_dev_raw.iloc[:, 0]])


    print("y_train_raw shape:", y_train_raw.shape)
    print("y_train (encoded) shape:", y_train.shape)

    clf = get_model("classification", model_name, n_estimators=400)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    assert len(x_train) == len(y_train), \
        f"Length mismatch: X={len(x_train)}  Y={len(y_train)}"
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_dev)

    f1_micro = f1_score(y_dev, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_dev, y_pred, average="macro", zero_division=0)
    print(f"Micro F1: {f1_micro:.4f} | Macro F1: {f1_macro:.4f}")

    # Coverage logging
    pred_counts = np.sum(y_pred, axis=0)
    never_predicted = [encoder.ind_to_label[i] for i in range(encoder.num_labels) if pred_counts[i] == 0]
    print(f"Never predicted labels ({len(never_predicted)}): {never_predicted[:5]}...")

    # Convert predictions back to label strings
    y_pred_str = [encoder.from_binary_vector(row) for row in y_pred]

    # Format as stringified lists for CSV output
    df_out = pd.DataFrame({
        'predicted_labels': [str(labs) for labs in y_pred_str]
    })

    # Optional: include index or dev IDs
    # if 'id-hushed_internalpatientid' in y_dev_raw.columns:
    #     df_out['patient_id'] = y_dev_raw['id-hushed_internalpatientid'].values


    # Save to CSV — include model and seed in filename
    df_out.to_csv(os.path.join(output_dir, f"predictions_{model_name}_seed{seed}.csv"), index=False)
    print(f"Saved predictions to outputs/predictions_{model_name}_seed{seed}.csv")

    # --- Predict on TEST set ---
    test_path = "../../data/train_test_splits/test.feats.csv"
    x_test = pd.read_csv(test_path, low_memory=False, dtype=str)
    x_test_cleaned = clean_data(remove_cols(x_test))

    # Predict
    y_test_pred = clf.predict(x_test_cleaned)
    y_test_pred_str = [encoder.from_binary_vector(row) for row in y_test_pred]

    # Save to CSV
    df_test_out = pd.DataFrame({
        'predicted_labels': [str(labs) for labs in y_test_pred_str]
    })


    df_test_out.to_csv(os.path.join(output_dir, f"test_predictions_{model_name}_seed{seed}.csv"), index=False)
    print(f"Saved test set predictions to outputs/test_predictions_{model_name}_seed{seed}.csv")
