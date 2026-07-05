import sys
import os

APP_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(APP_DIR, "..", "src"))

import streamlit as st
import joblib

from text_cleaning import clean_text

st.title("Détection de tweets suspects")
st.write("Écris un tweet ci-dessous pour savoir si le modèle le considère comme suspect.")

# le modele est aussi copie ici (avec Git LFS) pour que le deploiement cloud
# puisse le charger, vu que models/ est gere par DVC (stockage local)
vectorizer = joblib.load(os.path.join(APP_DIR, "model", "vectorizer.joblib"))
model = joblib.load(os.path.join(APP_DIR, "model", "model.joblib"))

tweet = st.text_area("Tweet")

if st.button("Prédire"):
    if tweet.strip() == "":
        st.warning("Écris un tweet avant de lancer la prédiction.")
    else:
        texte_nettoye = clean_text(tweet)
        X = vectorizer.transform([texte_nettoye])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        if prediction == 1:
            st.error(f"Tweet suspect (probabilité : {proba[1]:.2%})")
        else:
            st.success(f"Tweet non suspect (probabilité : {proba[0]:.2%})")
