"""Stage DVC 1 : prétraitement des tweets (Partie 1 du sujet).

Entrée : data/raw/tweets_suspect.csv
Sorties : data/processed/train.csv, data/processed/test.csv
"""
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from text_cleaning import clean_text

RAW_PATH = "data/raw/tweets_suspect.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

params = yaml.safe_load(open("params.yaml"))["preprocess"]

df = pd.read_csv(RAW_PATH)

# on enlève les lignes vides et les doublons
df = df.dropna(subset=["message", "label"])
df = df.drop_duplicates(subset=["message"])

df["clean_text"] = df["message"].apply(clean_text)

# après nettoyage, certains tweets peuvent devenir vides (ex: juste un lien)
df = df[df["clean_text"].str.len() > 0]

train_df, test_df = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=df["label"],
)

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("train :", len(train_df), "lignes")
print("test :", len(test_df), "lignes")
print(train_df["label"].value_counts(normalize=True))
