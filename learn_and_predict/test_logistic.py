from src.load_and_split import load_and_split_part1
from src.process_data import remove_cols, clean_data
from src.models import get_model
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

train_feats = "../data/train_test_splits/train.feats.csv"
labels_0 = "../data/train_test_splits/train.labels.0.csv"

model_name = "logistic"

print("\n Testing K-Nearest Neighbors (multi-label classification)...")

for seed in [0, 1, 2, 3, 4]:
    print(f"\nSeed: {seed}")
    x_train, x_dev, y_train_raw, y_dev_raw, encoder = load_and_split_part1(train_feats, labels_0, seed=seed)

    x_train = clean_data(remove_cols(x_train))
    x_dev = clean_data(remove_cols(x_dev))

    y_train = np.array([encoder.to_binary_vector(eval(row)) for row in y_train_raw.iloc[:, 0]])
    y_dev = np.array([encoder.to_binary_vector(eval(row)) for row in y_dev_raw.iloc[:, 0]])

    clf = get_model("classification", "logistic")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_dev)

    f1_micro = f1_score(y_dev, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_dev, y_pred, average="macro", zero_division=0)
    print(f"Micro F1: {f1_micro:.4f} | Macro F1: {f1_macro:.4f}")

    pred_counts = np.sum(y_pred, axis=0)
    never_predicted = [encoder.ind_to_label[i] for i in range(encoder.num_labels) if pred_counts[i] == 0]
    print(f"❗ Never predicted labels ({len(never_predicted)}): {never_predicted[:5]}...")

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
    df_out.to_csv(f"predictions_{model_name}_seed{seed}.csv", index=False)
    print(f"📄 Saved predictions to predictions_{model_name}_seed{seed}.csv")