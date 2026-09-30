import pandas as pd
from preprocess import clean_text

df = pd.read_csv("../data/sms.tsv", sep="\t", header=None, names=["label", "message"])

for i in range(5):
    original = df["message"][i]
    cleaned = clean_text(original)
    print(f"ORIGINAL: {original}")
    print(f"CLEANED:  {cleaned}")
    print()