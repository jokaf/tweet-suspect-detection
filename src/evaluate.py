"""Stage DVC 3 : évaluation du modèle (Parties 5 et 6).

Sorties : reports/metrics.json, reports/figures/confusion_matrix.png,
          reports/figures/roc_curve.png
"""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TEST_PATH = "data/processed/test.csv"
VECTORIZER_PATH = "models/vectorizer.joblib"
MODEL_PATH = "models/model.joblib"
METRICS_PATH = "metrics/metrics.json"

df = pd.read_csv(TEST_PATH)
df["clean_text"] = df["clean_text"].fillna("")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

X_test = vectorizer.transform(df["clean_text"])
y_test = df["label"]

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)

# matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["non suspect", "suspect"])
disp.plot(cmap="Blues")
plt.title("Matrice de confusion")
plt.savefig("metrics/confusion_matrix.png")
plt.close()

# courbe ROC
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Courbe ROC")
plt.savefig("metrics/roc_curve.png")
plt.close()
