import os
import re
from pathlib import Path
import pandas as pd
import pydicom


def extract_and_split_descriptions(root_dir):
    """
    Scans DICOM files, extracts raw descriptions, splits them strictly
    by delimiters, and dynamically creates positional token columns.
    """
    records = []
    root_path = Path(root_dir)

    for study_dir in root_path.iterdir():
        if not study_dir.is_dir():
            continue

        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue

            # Read DICOM header from first file in folder
            dcm_files = list(series_dir.glob("*.dcm"))
            description = None

            if dcm_files:
                try:
                    dcm_data = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                    description = getattr(dcm_data, 'ScanningSequence', None)
                except Exception:
                    pass

            # Fallback to series directory name if header is absent
            if not description:
                description = series_dir.name

            # Split on underscore and hyphen delimiters, removing empty strings
            tokens = [f"{description}"]#[t.strip() for t in re.split(r'[_,-]', description) if t.strip()]

            record = {
                'study_id': study_dir.name,
                'series_id': series_dir.name,
                'raw_description': description,
                'num_tokens': len(tokens),
            }

            # Map positional tokens dynamically (token_1, token_2, etc.)
            for i, token in enumerate(tokens, start=1):
                record[f'token_{i}'] = token

            records.append(record)

    return pd.DataFrame(records)


def get_unique_dataset_values(df):
    """
    Finds all unique string tokens actually present across the dataset
    and their frequency of occurrence.
    """
    token_cols = [col for col in df.columns if col.startswith('token_')]

    # Flatten all token columns into a single series
    all_values = df[token_cols].melt()['value'].dropna()

    # Get exact counts of present values
    summary = all_values.value_counts().reset_index()
    summary.columns = ['dataset_value', 'frequency']
    return summary

# --- Usage ---
# 1. Process dataset
df = extract_and_split_descriptions('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection/train_series')

# 2. View per-series parsed data
print(df.head())

# 3. View all distinct values present across the dataset
unique_values = get_unique_dataset_values(df)
unique_values.to_csv('unique_values.csv', index=False)
print(unique_values)