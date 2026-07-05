import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import joblib

from text_cleaning import clean_text

st.title("Detection de tweets suspects")
st.write("Ecris un tweet ci-dessous pour savoir si le modele le considere comme suspect.")

vectorizer = joblib.load("models/vectorizer.joblib")
model = joblib.load("models/model.joblib")

tweet = st.text_area("Tweet")

if st.button("Predire"):
    if tweet.strip() == "":
        st.warning("Ecris un tweet avant de lancer la prediction.")
    else:
        texte_nettoye = clean_text(tweet)
        X = vectorizer.transform([texte_nettoye])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        if prediction == 1:
            st.error(f"Tweet suspect (probabilite : {proba[1]:.2%})")
        else:
            st.success(f"Tweet non suspect (probabilite : {proba[0]:.2%})")
