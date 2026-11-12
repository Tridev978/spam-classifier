import pandas as pd
df=pd.read_csv("data/spam.csv",encoding="latin-1")[["v1","v2"]]
df.columns=["label","message"]
print(df.head())
print(df["label"].value_counts())
print("Duplicates:",df.duplicated().sum())
print("Average length:\n",df.groupby("label")["message"].apply(lambda s:s.str.len().mean()))