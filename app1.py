import streamlit as st
import pandas as pd
import pickle
import random
import nltk
from textblob import Word
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

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

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess text (title)
def preprocess_text(text):
    # Tokenization using simple split
    tokens = text.split()

    # Remove stopwords
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    
    # Lemmatization using WordNetLemmatizer
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
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
st.markdown("""
    <style>
        body {
            background-color: #f4f4f9;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #3e4e57;
            font-size: 36px;
            text-align: center;
            margin-top: 50px;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #6c757d;
            margin-bottom: 40px;
        }
        .input-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 0 auto;
            max-width: 500px;
        }
        .input-box {
            font-size: 18px;
            padding: 12px;
            border-radius: 6px;
            width: 100%;
            border: 1px solid #ddd;
            margin-bottom: 20px;
        }
        .submit-btn {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            width: 100%;
        }
        .submit-btn:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            font-size: 24px;
            text-align: center;
            font-weight: bold;
            color: #007bff;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the App
st.markdown("<h1 class='title'>Movie Rating Prediction</h1>", unsafe_allow_html=True)

# Subtitle
st.markdown("<h2 class='subtitle'>Enter a Bollywood movie name, and get its predicted rating category.</h2>", unsafe_allow_html=True)

# Input Box for Movie Title
movie_title = st.text_input("Enter Movie Title", key='movie_title', help="Type the name of the Bollywood movie", placeholder="e.g., Khel Khel Mein")

# Load the model and data
df = load_data()
model = load_model()

if movie_title:
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model)
    st.markdown(f"<p class='result'>Predicted category for <b>{movie_title}</b>: {predicted_category}</p>", unsafe_allow_html=True)

# Button to Trigger the Prediction
if st.button("Get Rating Category", use_container_width=True):
    if movie_title:
        predicted_category = predict_rating_category_from_dataset(movie_title, df, model)
        st.markdown(f"<p class='result'>Predicted category for <b>{movie_title}</b>: {predicted_category}</p>", unsafe_allow_html=True)
    else:
        st.error("Please enter a movie title!")
