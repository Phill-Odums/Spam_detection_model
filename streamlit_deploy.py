import streamlit as st
import joblib

# Load assets
model = joblib.load("C:/Users/phill/.spyder-py3/projects/Spam_detection/model.joblib")
vectorizer = joblib.load("C:/Users/phill/.spyder-py3/projects/Spam_detection/vectorizer.joblib")

# App UI
st.title("Spam Detector")
user_input = st.text_area("Enter message:")

if st.button("Check"):
    if user_input:
        try:
            X = vectorizer.transform([user_input])  # 2D input
            pred = int(model.predict(X)[0])  # Explicitly convert to int
            proba = float(model.predict_proba(X)[0][1])  # Explicitly convert to float
            
            if pred == 1:
                st.error(f"🚨 SPAM ({proba:.0%} confidence)")
            else:
                st.success(f"✅ NOT SPAM ({1-proba:.0%} confidence)")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Enter text first.")