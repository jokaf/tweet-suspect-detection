"""Stage DVC 2 : représentation TF-IDF + entraînement du modèle (Parties 3 et 4).

Entrée : data/processed/train.csv
Sorties : models/vectorizer.joblib, models/model.joblib
"""
import yaml
import joblib
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

TRAIN_PATH = "data/processed/train.csv"
VECTORIZER_PATH = "models/vectorizer.joblib"
MODEL_PATH = "models/model.joblib"

params = yaml.safe_load(open("params.yaml"))["train"]

# suivi des experiences en plus de DVC (bonus)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("tweet-suspect-detection")
mlflow.start_run()
mlflow.log_params(params)

df = pd.read_csv(TRAIN_PATH)
df["clean_text"] = df["clean_text"].fillna("")

X_text = df["clean_text"]
y = df["label"]

# on transforme le texte en vecteurs TF-IDF
vectorizer = TfidfVectorizer(
    max_features=params["max_features"],
    ngram_range=(1, params["ngram_max"]),
)
X = vectorizer.fit_transform(X_text)

# gestion du déséquilibre des classes (Partie 4)
strategy = params["imbalance_strategy"]
class_weight = None

if strategy == "class_weight":
    class_weight = "balanced"
elif strategy == "smote":
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=params["random_state"])
    X, y = smote.fit_resample(X, y)

# choix du modèle selon params.yaml
model_name = params["model"]

if model_name == "logistic_regression":
    model = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=params["random_state"])
elif model_name == "naive_bayes":
    model = MultinomialNB()
elif model_name == "random_forest":
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        class_weight=class_weight,
        random_state=params["random_state"],
    )
else:
    raise ValueError("modèle inconnu dans params.yaml : " + model_name)

model.fit(X, y)

joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(model, MODEL_PATH)

mlflow.log_metric("nombre_exemples", X.shape[0])
mlflow.end_run()

print("modèle entraîné :", model_name)
print("stratégie de rééquilibrage :", strategy)
print("nombre d'exemples utilisés :", X.shape[0])
