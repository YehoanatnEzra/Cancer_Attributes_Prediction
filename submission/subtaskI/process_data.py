import numpy as np
import pandas as pd
import re

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
        'אבחנה-Nodes exam',
        'אבחנה-Surgery date1',
        'אבחנה-Surgery date2',
        'אבחנה-Surgery date3',
        'אבחנה-Tumor depth',
        'אבחנה-Tumor width'
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

    # Apply HER2 mapping to raw column and create standardized HER2 feature
    data_frame['אבחנה-Her2'] = data_frame['אבחנה-Her2'].apply(map_her2)

    # --------
    # Apply Histological diagnosis mapping:
    vectors = data_frame['אבחנה-Histological diagnosis'].apply(map_histological_to_vector)
    # Convert list of vectors into dataframe columns
    vector_df = pd.DataFrame(vectors.tolist(), index=data_frame.index,columns=[
        'HistDiag_Ductal', 'HistDiag_Lobular', 'HistDiag_DCIS',
        'HistDiag_Mixed', 'HistDiag_Special', 'HistDiag_Benign', 'HistDiag_Other'
    ])
    # Concatenate back to main dataframe
    data_frame = pd.concat([data_frame, vector_df], axis=1)
    # Original text column dropped
    data_frame = data_frame.drop(columns=['אבחנה-Histological diagnosis'])
    # --------

    # positive nodes col. fill empty cells to 0.
    data_frame['אבחנה-Positive nodes'] = data_frame['אבחנה-Positive nodes'].fillna(0).astype(float)

    # ---- name 1 ----
    # Apply mapping
    vectors = data_frame['אבחנה-Surgery name1'].apply(map_surgery_name1)
    # Expand to dataframe
    vector_df = pd.DataFrame(vectors.tolist(), index=data_frame.index,columns=[
        'Surg_Mastectomy', 'Surg_Lumpectomy', 'Surg_LymphNodes', 'Surg_OtherBreast', 'Surg_Other'
    ])
    # Concatenate
    data_frame = pd.concat([data_frame, vector_df], axis=1)
    # Drop original
    data_frame = data_frame.drop(columns=['אבחנה-Surgery name1'])
    # --------

    # ---- name 2 ----
    # Apply mapping
    vectors = data_frame['אבחנה-Surgery name2'].apply(map_surgery_name2)
    # Expand to dataframe
    vector_df = pd.DataFrame(vectors.tolist(), index=data_frame.index,columns=[
        'Surg2_Mastectomy', 'Surg2_Lumpectomy', 'Surg2_LymphNodes', 'Surg2_OtherBreast', 'Surg2_Other'
    ])
    # Concatenate
    data_frame = pd.concat([data_frame, vector_df], axis=1)
    # Drop original
    data_frame = data_frame.drop(columns=['אבחנה-Surgery name2'])
    # --------

    # ---- name 3 ----
    # Apply mapping
    vectors = data_frame['אבחנה-Surgery name3'].apply(map_surgery_name3)
    # Expand to dataframe
    vector_df = pd.DataFrame(vectors.tolist(), index=data_frame.index,columns=[
        'Surg3_Mastectomy', 'Surg3_Lumpectomy', 'Surg3_LymphNodes', 'Surg3_OtherBreast', 'Surg3_Other'
    ])
    # Concatenate
    data_frame = pd.concat([data_frame, vector_df], axis=1)
    # Drop original
    data_frame = data_frame.drop(columns=['אבחנה-Surgery name3'])
    # --------

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
        .astype(int)
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
        .astype(int)
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

    # Apply mapping
    data_frame['אבחנה-er'] = data_frame['אבחנה-er'].apply(map_er)

    # Apply mapping
    data_frame['אבחנה-pr'] = data_frame['אבחנה-pr'].apply(map_pr)

    return data_frame


def map_her2(value):
    """
    Map raw HER2 test results to standardized categories:
    0=Unknown, 1=Negative, 2=Equivocal, 3=Positive.
    """
    if pd.isna(value):
        return 0  # Unknown

    value = str(value).lower().strip()

    # Remove common noise
    value = re.sub(r'[^a-z0-9\+\- ]', '', value)

    # Handle obvious positives
    if any(pos in value for pos in
           ['+3', '3+', 'pos +3', 'positive by ihc and fish', 'amplified', 'fish amplified', 'fish+', 'pos +']):
        return 3

    if any(pos in value for pos in ['positive', 'positiv']):
        return 3

    # Handle obvious negatives
    if any(neg in value for neg in
           ['negative', 'neg ', 'neg.', 'neg', 'negetive', 'negativa', 'negative by fish', 'her2/neu negative',
            'fish non amplified', 'not amplified', 'non amplified', '0']):
        return 1

    # Handle equivocal (borderline)
    if any(eq in value for eq in ['+2', '2+', 'equivocal', 'intermediate', 'borderline', 'indeterm']):
        return 2

    # Handle explicit zero values or noise values
    if value in ['-', '', 'nan', 'none', 'not done', 'pending', '?', 'nd', 'nfg', 'nef', 'akhah', 'akhkh', 'meg',
                 'heg']:
        return 0

    # If nothing matched, assign Unknown (0)
    return 0

def map_histological_to_vector(value):
    """
    Map Histological Diagnosis to one-hot vector: [DUCTAL, LOBULAR, DCIS, MIXED, SPECIAL, BENIGN, OTHER]
    """
    if pd.isna(value):
        return [0, 0, 0, 0, 0, 0, 1]

    value = str(value).strip().upper()

    # DCIS first
    if any(x in value for x in ['CARCINOMA IN SITU', 'INTRADUCTAL']):
        return [0, 0, 1, 0, 0, 0, 0]

    # Lobular next
    if 'LOBULAR INFILTRATING' in value or 'LOBULAR CARCINOMA IN SITU' in value:
        return [0, 1, 0, 0, 0, 0, 0]

    # Mixed
    if 'DUCTAL AND LOBULAR' in value or 'INTRADUCT AND LOBULAR' in value:
        return [0, 0, 0, 1, 0, 0, 0]

    # Benign tumors
    if any(x in value for x in ['BENIGN', 'FIBROADENOMA', 'ADENOMA', 'PHYLLODES TUMOR BENIGN',
                                 'PHYLLODES TUMOR NOS']):
        return [0, 0, 0, 0, 0, 1, 0]

    # Special types
    if any(x in value for x in ['TUBULAR', 'MEDULLARY', 'MUCINOUS', 'NEUROENDOCRINE',
                                 'PAPILLARY', 'INFLAMMATORY', 'APOCRINE', 'INTRACYSTIC',
                                 'COMEDOCARCINOMA']):
        return [0, 0, 0, 0, 1, 0, 0]

    # Finally ductal (only if none of the above matched)
    if any(x in value for x in ['INFILTRATING DUCT', 'DUCTULAR']):
        return [1, 0, 0, 0, 0, 0, 0]

    # Default OTHER
    return [0, 0, 0, 0, 0, 0, 1]

def map_surgery_name1(value):
    """
    Map 'אבחנה-Surgery name1' into one-hot vector of 5 clinical surgery groups:
    [MASTECTOMY, LUMPECTOMY, LYMPH_NODES, OTHER_BREAST, OTHER]
    """
    if pd.isna(value):
        return [0, 0, 0, 0, 1]  # OTHER

    value = str(value).strip().upper()

    if any(x in value for x in [
        'MASTECTOMY', 'RADICAL MODIFIED MASTECTOMY', 'SIMPLE MASTECTOMY',
        'EXTENDED MASTECTOMY', 'SUBTOTAL MASTECTOMY', 'UNILATERAL', 'BILATERAL'
    ]):
        return [1, 0, 0, 0, 0]

    if any(x in value for x in [
        'LUMPECTOMY', 'LOCAL EXC', 'QUADRANTECTOMY'
    ]):
        return [0, 1, 0, 0, 0]

    if any(x in value for x in [
        'AXILLARY', 'LYMPH NODE', 'SENTINEL NODE', 'SOLITARY LYMPH', 'EXCISION OF AXILLARY'
    ]):
        return [0, 0, 1, 0, 0]

    if any(x in value for x in [
        'BIOPSY OF BREAST', 'OPEN BIOPSY', 'ECTOPIC BREAST'
    ]):
        return [0, 0, 0, 1, 0]

    return [0, 0, 0, 0, 1]

def map_surgery_name2(value):
    """
    Map 'אבחנה-Surgery name2' into one-hot vector of 5 clinical surgery groups:
    [MASTECTOMY, LUMPECTOMY, LYMPH_NODES, OTHER_BREAST, OTHER]
    """
    if pd.isna(value):
        return [0, 0, 0, 0, 1]  # OTHER

    value = str(value).strip().upper()

    if any(x in value for x in [
        'MASTECTOMY', 'SIMPLE MASTECTOMY', 'SUBTOTAL MASTECTOMY',
        'EXTENDED SIMPLE MASTECTOMY', 'UNILATERAL', 'BILATERAL'
    ]):
        return [1, 0, 0, 0, 0]

    if any(x in value for x in [
        'LUMPECTOMY', 'LOCAL EXC', 'QUADRANTECTOMY'
    ]):
        return [0, 1, 0, 0, 0]

    if any(x in value for x in [
        'AXILLARY', 'LYMPH NODE', 'SENTINEL NODE'
    ]):
        return [0, 0, 1, 0, 0]

    if any(x in value for x in [
        'BIOPSY OF BREAST', 'OPEN BIOPSY'
    ]):
        return [0, 0, 0, 1, 0]

    return [0, 0, 0, 0, 1]

def map_surgery_name3(value):
    """
    Map 'אבחנה-Surgery name3' into one-hot vector of 5 clinical surgery groups:
    [MASTECTOMY, LUMPECTOMY, LYMPH_NODES, OTHER_BREAST, OTHER]
    """
    if pd.isna(value):
        return [0, 0, 0, 0, 1]  # OTHER

    value = str(value).strip().upper()

    if 'MASTECTOMY' in value:
        return [1, 0, 0, 0, 0]

    if any(x in value for x in ['LUMPECTOMY', 'LOCAL EXC']):
        return [0, 1, 0, 0, 0]

    if 'LYMPH NODE' in value or 'AXILLARY' in value:
        return [0, 0, 1, 0, 0]

    return [0, 0, 0, 0, 1]


def map_er(value):
    """
    Map 'אבחנה-er' to 0=Unknown, 1=Negative, 2=Equivocal, 3=Positive.
    """
    if pd.isna(value):
        return 0

    value = str(value).strip().upper()

    # First handle clear negatives
    if any(x in value for x in ['NEG', 'NETGATIVE', 'NEGATIVE', 'שלילי']):
        return 1

    # Handle clear positives
    if any(x in value for x in
           ['POS', 'POSITIVE', 'חיובי', 'STRONG', '+++', '3+', '+3', '+4', '100%', '95%', '90%', '80%', '70%', '60%']):
        return 3

    # Handle weak or equivocal
    if any(x in value for x in ['WEAK', 'INTERM', 'INTERMEDIATE', 'EQUIV', '1+', '2+', 'WEAKLY']):
        return 2

    # Handle empty strings or garbage
    if value in ['', '-', '(', '?', '#NAME?']:
        return 0

    # Default: treat as unknown
    return 0

def map_pr(value):
    """
    Map 'אבחנה-pr' to 0=Unknown, 1=Negative, 2=Equivocal, 3=Positive.
    """
    if pd.isna(value):
        return 0

    value = str(value).strip().upper()

    # Clean common garbage
    if value in ['', '-', '_', '?', '#NAME?']:
        return 0

    # Negatives
    if any(x in value for x in ['NEG', 'NEGATIVE', 'שלילי', 'NEG/NEG', 'NEGNEG']):
        return 1

    # Positives (strong)
    if any(x in value for x in [
        'POS', 'POSITIVE', 'חיובי', '+++', '3+', '+3', '+4',
        '100%', '95%', '90%', '80%', '70%', '60%', 'STRONG'
    ]):
        return 3

    # Weak or equivocal
    if any(x in value for x in ['WEAK', 'INTERM', 'INTERMEDIATE', 'EQUIV', '1+', '2+', 'WEAKLY']):
        return 2

    # Default unknown
    return 0


if __name__ == '__main__':
    df_train = pd.read_csv("../data/train_test_splits/train.feats.csv", dtype=str)
    df_train = remove_cols(df_train)
    df_train = clean_data(df_train)

    # ---
    # to see it clear in a new CSV:
    # df_train.to_csv("train_cleaned.csv", index=False)