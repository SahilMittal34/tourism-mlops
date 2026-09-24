import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = os.path.join("data", "tourism.csv")
OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INPUT)
df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
train, test = train_test_split(df, test_size=0.20, stratify=df["ProdTaken"], random_state=42)
train.to_csv(os.path.join(OUT, "train.csv"), index=False)
test.to_csv(os.path.join(OUT, "test.csv"), index=False)
print(f"Prepared train shape: {train.shape}")
print(f"Prepared test shape: {test.shape}")
