import streamlit as st
import joblib
@st.cache_resource
def load_model():
    return joblib.load("model/spam_model.joblib")

model=load_model()
st.title("SMS Spam Detector")
st.write("Paste a message and the model will tell you if it looks like spam.")
examples = {
    "Spam example": "WINNER!! You have been selected for a £900 prize. Call 09061701461 now.",
    "Normal example": "Are we still meeting at 5 today?",
}
choice = st.selectbox("Or pick an example:", ["(none)"] + list(examples))
text = examples.get(choice, "")  
text=st.text_area("Message:",value=text,height=150)

if st.button("Check"):
    if not text.strip():
        st.warning("Please type a message first.")
    else:
        pred=model.predict([text])[0]
        prob=model.predict_proba([text])[0][1]
        if pred==1:
            st.error(f"🚨 Spam (confidence: {prob:.0%})")
        else:
            st.success(f"Not spam(spam: probability: {prob:.0%})")
        st.progress(float(prob))
        st.caption(f"Spam probability: {prob:.1%}")
