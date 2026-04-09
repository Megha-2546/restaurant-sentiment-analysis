import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

st.title("🍽️ Restaurant Review Sentiment Analysis")

input_text = st.text_area("Enter your review:")

if st.button("Predict"):
    processed = preprocess(input_text)
    vector = cv.transform([processed]).toarray()
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("😊 Positive Review")
    else:
        st.error("😞 Negative Review")