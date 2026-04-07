import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("AI Sentiment Analyzer")

user_input = st.text_area("Enter your movie review:")

if st.button("Predict"):
    prediction = model.predict(vectorizer.transform([user_input]))

    if prediction[0] == "positive":
        st.success("😊 Positive Review")
    else:
        st.error("😡 Negative Review")
