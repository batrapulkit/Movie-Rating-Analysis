import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from textblob import Word
from nltk.corpus import stopwords
import random

# Load the pre-trained model and dataset
@st.cache_resource
def load_model():
    # Load the pre-trained sentiment model
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_data
def load_data():
    # Replace with the correct path to your dataset
    df = pd.read_csv('movies.csv')
    
    # Function to replace zero ratings with random values between 5.1 and 7
    def replace_zero_ratings_with_random_value(rating):
        if rating == 0:
            return random.uniform(5.1, 7)  # Random value between 5.1 and 7
        else:
            return rating
    
    # Apply the function to the 'rating' column
    df['rating'] = df['rating'].apply(replace_zero_ratings_with_random_value)

    # Drop duplicates based on 'title', keeping the last occurrence
    df = df.drop_duplicates(subset='title', keep='last')
    
    return df

# Preprocess text (title)
def preprocess_text(text):
    # Tokenization using simple split, no external library needed
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    
    # Lemmatization with TextBlob (it performs lemmatization)
    tokens = [Word(word).lemmatize() for word in tokens]
    
    return ' '.join(tokens)

# Function to predict the rating category for a movie using the movie title
def predict_rating_category_from_dataset(title, df, model):
    # Preprocess the title (ensure the same preprocessing is applied to the input text)
    processed_title = preprocess_text(title)

    # Check if the movie title exists in the dataset
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    # If movie is found in the dataset
    if not movie_data.empty:
        # Extract the rating from the dataset and convert it to a numeric value
        rating = pd.to_numeric(movie_data.iloc[0]['rating'], errors='coerce')  # Convert to numeric
        
        # If rating is not NaN (i.e., valid rating value)
        if not pd.isna(rating):
            # Predict the category based on the rating
            if rating >= 7:
                return "Good"
            elif 5 <= rating < 7:
                return "Neutral"
            else:
                return "Bad"
        else:
            return "Invalid rating data"
    else:
        return "Movie not found in dataset"

# Streamlit interface
st.title("Movie Rating Prediction")

movie_title = st.text_input("Enter Movie Title")

# Load the model and data
df = load_data()
model = load_model()

if movie_title:
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model)
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")
