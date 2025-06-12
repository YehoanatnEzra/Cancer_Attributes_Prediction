import pandas as pd

def remove_cols(data_frame):
    # Remove irrelevant columns not used for modeling
    data_frame = data_frame.drop([
        ' Form Name',
        ' Hospital',
        'User Name',
        'id-hushed_internalpatientid',
        'surgery before or after-Activity date',
        'surgery before or after-Actual activity',
        'אבחנה-Diagnosis date',
        'אבחנה-Her2',
        'אבחנה-Histological diagnosis',
        'אבחנה-Nodes exam',
        'אבחנה-Positive nodes',
        'אבחנה-Surgery date1',
        'אבחנה-Surgery date2',
        'אבחנה-Surgery date3',
        'אבחנה-Surgery name1',
        'אבחנה-Surgery name2',
        'אבחנה-Surgery name3',
        'אבחנה-Tumor depth',
        'אבחנה-Tumor width',
        'אבחנה-er',
        'אבחנה-pr'
    ], axis=1)

    return data_frame


def clean_data(data_frame):
    # Convert patient age to float (medical age may have decimal precision)
    data_frame['אבחנה-Age'] = data_frame['אבחנה-Age'].astype(float)

    # Map 'Basic stage' clinical/pathological types to integer codes
    mapping = {
        'p - Pathological': 2,
        'c - Clinical': 1,
        'Null': 0,
        'r - Reccurent': 3
    }
    data_frame['אבחנה-Basic stage'] = (
        data_frame['אבחנה-Basic stage']
        .map(mapping)
        .fillna(0)
        .astype(int)
    )

    # Map histopathological grading levels to ordinal scale
    mapping = {
        'Null': 0,
        'GX - Grade cannot be assessed': 0,
        'G1 - Well Differentiated': 1,
        'G2 - Modereately well differentiated': 2,
        'G3 - Poorly differentiated': 3,
        'G4 - Undifferentiated': 4
    }
    data_frame['אבחנה-Histopatological degree'] = (
        data_frame['אבחנה-Histopatological degree']
        .map(mapping)
        .fillna(0)
        .astype(int)
    )

    # Standardize lymphovascular invasion presence (binary classification)
    data_frame['אבחנה-Ivi -Lymphovascular invasion'] = (
        data_frame['אבחנה-Ivi -Lymphovascular invasion']
        .replace({
            '0': 0, '+': 1, 'extensive': 1, 'yes': 1, '(+)': 1,
            'no': 0, '(-)': 0, 'none': 0, 'No': 0, 'not': 0, '-': 0,
            'NO': 0, 'neg': 0, 'MICROPAPILLARY VARIANT': 1
        })
        .where(
            data_frame['אבחנה-Ivi -Lymphovascular invasion'].isin([0, 1]), 0
        )
    )

    # Extract numeric Ki67 protein percentage, clip upper bound to 100%
    data_frame['אבחנה-KI67 protein'] = (
        data_frame['אבחנה-KI67 protein']
        .str.extract(r'(\d+)')[0]
        .fillna(0)
        .astype(int)
        .clip(upper=100)
    )

    # Map lymphatic penetration levels to ordinal coding
    mapping = {
        'Null': 0,
        'L0 - No Evidence of invasion': 0,
        'LI - Evidence of invasion': 1,
        'L1 - Evidence of invasion of superficial Lym.': 2,
        'L2 - Evidence of invasion of depp Lym.': 3
    }
    data_frame['אבחנה-Lymphatic penetration'] = (
        data_frame['אבחנה-Lymphatic penetration']
        .map(mapping)
        .fillna(0)
        .astype(int)
    )

    # Extract numeric metastasis indicator (M in TNM staging)
    data_frame['אבחנה-M -metastases mark (TNM)'] = (
        data_frame['אבחנה-M -metastases mark (TNM)']
        .str.extract(r'(\d+)')[0]
        .fillna(0)
        .astype(int)
    )

    # Map margin type (surgery outcome) to binary indicator
    mapping = {
        "נקיים": 0,
        "ללא": 0,
        "נגועים": 1
    }
    data_frame['אבחנה-Margin Type'] = (
        data_frame['אבחנה-Margin Type']
        .map(mapping)
        .fillna(0)
        .astype('Int64')
    )

    # Extract numeric N stage (lymph node involvement)
    data_frame['אבחנה-N -lymph nodes mark (TNM)'] = (
        data_frame['אבחנה-N -lymph nodes mark (TNM)']
        .str.extract(r'(\d+)')[0]
        .fillna(0)
        .astype(int)
    )

    # Map tumor side (laterality) to integer codes
    mapping = {
        "שמאל": 1,
        "ימין": 1,
        "דו צדדי": 2
    }
    data_frame['אבחנה-Side'] = (
        data_frame['אבחנה-Side']
        .map(mapping)
        .fillna(0)
        .astype('Int64')
    )

    # Extract numeric overall stage from string representation
    data_frame['אבחנה-Stage'] = (
        data_frame['אבחנה-Stage']
        .str.extract(r'(\d+)')[0]
        .fillna(0)
        .astype(int)
    )

    # Clean surgery sum (number of surgical procedures)
    data_frame['אבחנה-Surgery sum'] = (
        data_frame['אבחנה-Surgery sum']
        .fillna(0)
        .astype(float)
    )

    # Extract numeric T stage (tumor size indicator)
    data_frame['אבחנה-T -Tumor mark (TNM)'] = (
        data_frame['אבחנה-T -Tumor mark (TNM)']
        .str.extract(r'(\d)')[0]
        .fillna(0)
        .astype(int)
    )

    return data_frame




if __name__ == '__main__':
    df_train = pd.read_csv("data/train_test_splits/train.feats.csv", dtype=str)
    df_train = remove_cols(df_train)
    df_train = clean_data(df_train)

    # to see it clear in a new CSV:
    df_train.to_csv("train_cleaned.csv", index=False)