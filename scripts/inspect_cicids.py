import glob
import pandas as pd

files = sorted(glob.glob("data/raw/cicids2017/*.parquet"))
first = pd.read_parquet(files[0])
print(first.columns.tolist())
print("Protocol" in first.columns)
labels = pd.concat([pd.read_parquet(f, columns=["Label"]) for f in files])
print(len(labels))
print(labels["Label"].value_counts())