import os
import json

import streamlit as st

st.set_page_config(page_title="Dashboard")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard_data")

st.title("Dashboard du modèle")
st.write("Cette page montre les résultats du modèle final (Random Forest), "
         "pour pouvoir suivre ses performances sans avoir à rouvrir les notebooks.")

with open(os.path.join(DATA_DIR, "metrics.json")) as f:
    metrics = json.load(f)

st.divider()
st.subheader("Métriques sur le jeu de test")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col2.metric("Precision", f"{metrics['precision']:.3f}")
col3.metric("Recall", f"{metrics['recall']:.3f}")
col4.metric("F1-score", f"{metrics['f1_score']:.3f}")
col5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

st.divider()
st.subheader("Distribution des classes")
st.caption("Le dataset est déséquilibré, environ 90% de tweets suspects et 10% non suspects.")
st.image(os.path.join(DATA_DIR, "class_distribution.png"))

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Matrice de confusion")
    st.image(os.path.join(DATA_DIR, "confusion_matrix.png"))
with col_b:
    st.subheader("Courbe ROC")
    st.image(os.path.join(DATA_DIR, "roc_curve.png"))
