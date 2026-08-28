import pandas as pd

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def split_train_val(input_soft_csv='train_soft_labels.csv', orig_csv='train.csv'):
    # Load original train.csv to identify gold annotations
    df_orig = pd.read_csv(orig_csv)
    df_soft = pd.read_csv(input_soft_csv)

    # Boolean mask for the 58 official radiologist-labeled rows
    gold_mask = df_orig[TARGET_COLS].notna().all(axis=1)

    # 1. Validation Set (58 Gold Studies - Binary ground truth)
    val_gold_df = df_orig[gold_mask].copy()

    # 2. Training Set (4,349 Studies - Soft targets from report parsing)
    train_soft_df = df_soft[~gold_mask].copy()

    # Verify zero overlap between StudyInstanceUIDs
    train_uids = set(train_soft_df['StudyInstanceUID'])
    val_uids = set(val_gold_df['StudyInstanceUID'])
    assert len(train_uids.intersection(val_uids)) == 0, "Data leakage detected between train and val!"

    # Save to disk
    val_gold_df.to_csv('val_gold.csv', index=False)
    train_soft_df.to_csv('train_soft.csv', index=False)

    print(f"Validation Set (Gold): {len(val_gold_df)} studies -> saved to 'val_gold.csv'")
    print(f"Training Set (Pseudo-labeled): {len(train_soft_df)} studies -> saved to 'train_soft.csv'")

if __name__ == "__main__":
    split_train_val()