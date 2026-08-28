import pandas as pd
import numpy as np
import re
from sklearn.metrics import roc_auc_score

# 12 Target classes for RSNA Knee Abnormality Detection
TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Multilingual medical patterns (English, Spanish, Turkish, German, Greek)
PATTERNS = {
    'ACL': [r'\bacl\b', r'cruciate', r'cruzado', r'vorderes kreuzband', r'ön çapraz'],
    'MCL': [r'\bmcl\b', r'collateral', r'colateral', r'ic yan bag'],
    'Medial Meniscus': [r'medial meniscus', r'menisco interno', r'menisco medial', r'ic meniskus', r'innenmeniscus'],
    'Lateral Meniscus': [r'lateral meniscus', r'menisco externo', r'menisco lateral', r'dis meniskus',
                         r'aussenmeniscus'],
    'Medial OA': [r'medial.*osteoarthr', r'artros.*medial', r'gonartros.*medial',
                  r'medial.*compartment.*cartilage.*loss'],
    'Lateral OA': [r'lateral.*osteoarthr', r'artros.*lateral', r'gonartros.*lateral',
                   r'lateral.*compartment.*cartilage.*loss'],
    'PF OA': [r'patellofemoral', r'patela', r'rofula', r'pf.*oa', r'chondromalacia', r'condropatia'],
    'Effusion': [r'effusion', r'derrame', r'efüzyon', r'erguss', r'fluid'],
    'Synovitis': [r'synovitis', r'sinovitis', r'sinovit', r'synovialis'],
    "Baker's": [r'baker', r'popliteal cyst', r'quiste de baker', r'baker-zyste'],
    'Contusion': [r'contusion', r'bruise', r'edema oseo', r'kemik ödemi', r'bone marrow edema', r'marrow edema'],
    'Fracture': [r'fracture', r'fractura', r'kırık', r'kirik', r'fraktur']
}

NEGATIONS = [
    r'no\b', r'without', r'sin', r'ausente', r'normal', r'intact', r'intacto',
    r'salim', r'yok', r'denied', r'unremarkable', r'negativo', r'negative', r'kein', r'keine', r'preservad'
]

SEVERITY_KEYWORDS = ['tear', 'rupture', 'rotura', 'severe', 'yirtik', 'kopma', 'defect', 'loss', 'edema']


def score_report(text: str) -> dict:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {col: 0.15 for col in TARGET_COLS}  # Background prior for empty reports

    clean = text.replace('İ', 'i').replace('I', 'ı').lower()
    res = {}

    for col in TARGET_COLS:
        p_list = PATTERNS[col]
        found = False
        negated = False
        severe = False

        for p in p_list:
            matches = list(re.finditer(p, clean))
            if matches:
                found = True
                for m in matches:
                    start, end = m.start(), m.end()
                    # 80-character local context window
                    ctx = clean[max(0, start - 40): min(len(clean), end + 40)]

                    if any(re.search(r'\b' + n, ctx) for n in NEGATIONS):
                        negated = True
                    if any(w in ctx for w in SEVERITY_KEYWORDS):
                        severe = True
                break

        # Soft Probability Target Assignments
        if not found:
            res[col] = 0.15  # Unmentioned background prior
        elif negated and not severe:
            res[col] = 0.05  # Explicitly negative finding
        elif severe and not negated:
            res[col] = 0.90  # High-grade positive finding
        else:
            res[col] = 0.50  # Equivocal / mild finding

    return res


def execute_labeling_and_split(input_csv='train.csv'):
    # 1. Load Original Dataset
    df = pd.read_csv(input_csv)
    gold_mask = df[TARGET_COLS].notna().all(axis=1)

    print(f"Total Studies: {len(df)}")
    print(f"Official Gold Studies (Val): {gold_mask.sum()}")
    print(f"Unannotated Report Studies (Train): {(~gold_mask).sum()}")

    # 2. Extract Soft Targets for Unlabeled Training Reports
    output_df = df.copy()
    parsed_scores = df.loc[~gold_mask, 'Report'].apply(score_report).apply(pd.Series)

    for col in TARGET_COLS:
        output_df.loc[~gold_mask, col] = parsed_scores[col]

    # 3. Save Full Dataset with Soft Pseudo-Labels
    output_df.to_csv('train_soft_labels.csv', index=False)

    # 4. Generate Distinct Train/Val Files
    val_gold_df = df[gold_mask].copy()  # Exact binary 0/1 annotations
    train_soft_df = output_df[~gold_mask].copy()  # Soft probability pseudo-labels

    val_gold_df.to_csv('val_gold.csv', index=False)
    train_soft_df.to_csv('train_soft.csv', index=False)

    # 5. Benchmark Macro ROC-AUC against 58 Gold Studies
    gold_preds = val_gold_df['Report'].apply(score_report).apply(pd.Series)
    aucs = []

    print("\n--- Pseudo-Labeler Benchmark vs 58 Gold Ground-Truth Rows ---")
    for col in TARGET_COLS:
        auc = roc_auc_score(val_gold_df[col].astype(int), gold_preds[col])
        aucs.append(auc)
        print(f"{col:18s} AUC: {auc:.4f}")

    print(f"\nOverall Pseudo-Labeler Macro AUC: {np.mean(aucs):.4f}")
    print("\nFiles generated successfully:")
    print(" 1. 'train_soft.csv'     -> 4,349 pseudo-labeled studies for model training")
    print(" 2. 'val_gold.csv'       -> 58 expert binary studies for local validation")
    print(" 3. 'train_soft_labels.csv' -> Combined dataset (4,407 studies)")


if __name__ == "__main__":
    execute_labeling_and_split('train.csv')