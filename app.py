import streamlit as st
import pandas as pd
import pickle
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the pre-trained model and vectorizer
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

# Preprocess movie title text
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Function to predict the rating category for a movie using the model
def predict_rating_category_with_model(title, model, vectorizer, df):
    # Preprocess the input title
    processed_title = preprocess_text(title)
    title_features = vectorizer.transform([processed_title])
    
    # Get the movie's rating from the dataset
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        rating = movie_data.iloc[0]['rating']
        X_input = hstack([title_features, np.array([[rating]])])  # Combine title features and rating
        
        # Predict the rating category using the model
        predicted_category = model.predict(X_input)
        return predicted_category[0]
    else:
        return "Movie not found in dataset"

# Streamlit interface
st.title("Movie Rating Prediction")

# Load the model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

# Load dataset for movie look-up (for title-based predictions)
df = pd.read_csv('movies.csv')

movie_title = st.text_input("Enter Movie Title")

if movie_title:
    predicted_category = predict_rating_category_with_model(movie_title, model, vectorizer, df)
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")
