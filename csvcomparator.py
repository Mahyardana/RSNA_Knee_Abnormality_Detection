import pandas as pd

# Load the CSV files
df1 = pd.read_csv('labels_llm_gpt56sol.csv')
df2 = pd.read_csv('llm_generated_labels.csv')

# 1. Compare shape and column structure
print(f"File 1 Shape: {df1.shape}")
print(f"File 2 Shape: {df2.shape}\n")

col_diff1 = set(df1.columns) - set(df2.columns)
col_diff2 = set(df2.columns) - set(df1.columns)

if col_diff1 or col_diff2:
    print(f"Columns only in File 1: {col_diff1}")
    print(f"Columns only in File 2: {col_diff2}\n")
else:
    print("Column names match perfectly.\n")

# 2. Compare cell values (if structures align)
if df1.shape == df2.shape and list(df1.columns) == list(df2.columns):
    comparison = df1.compare(df2)
    if comparison.empty:
        print("The two CSV files are identical.")
    else:
        print("Differences found:")
        print(comparison)
else:
    print("Files differ in dimensions or columns. Align matching key/ID columns before running cell comparison.")