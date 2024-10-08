import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import movie_reviews

# Download movie reviews dataset
nltk.download('movie_reviews')

# Load movie reviews data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
np.random.shuffle(documents)

# Prepare the dataset
X = [" ".join(doc) for doc, category in documents]
y = [category for doc, category in documents]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for vectorization and model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Streamlit app layout
st.title("Sentiment Analysis App")
st.write("Enter a sentence to analyze its sentiment:")

# User input
user_input = st.text_area("Your text here:")

# Prediction
if st.button("Analyze Sentiment"):
    if user_input:
        prediction = model.predict([user_input])
        st.write(f"The sentiment of the input text is: **{prediction[0].capitalize()}**")
    else:
        st.write("Please enter some text for analysis.")

# Additional information
st.write("### How it works")
st.write("This app uses a Naive Bayes classifier trained on movie reviews to determine the sentiment of the input text.")
