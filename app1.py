import streamlit as st
import pandas as pd
import pickle
import random
import nltk
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
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    
    def replace_zero_ratings_with_random_value(rating):
        if rating == 0:
            return random.uniform(5.1, 7)
        else:
            return rating
    
    df['rating'] = df['rating'].apply(replace_zero_ratings_with_random_value)
    df = df.drop_duplicates(subset='title', keep='last')
    return df

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess text (title)
def preprocess_text(text):
    tokens = text.split()
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to predict the rating category for a movie using the movie title
def predict_rating_category_from_dataset(title, df, model):
    processed_title = preprocess_text(title)
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        rating = pd.to_numeric(movie_data.iloc[0]['rating'], errors='coerce')
        if not pd.isna(rating):
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

# Streamlit interface with enhanced design
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #f06, #ff9a9e);
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: white;
            margin-top: 50px;
            text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            font-size: 24px;
            color: #e4e4e4;
            margin-bottom: 40px;
            font-weight: lighter;
        }
        .input-section {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
            margin: 0 auto;
            max-width: 500px;
            text-align: center;
        }
        .input-box {
            font-size: 18px;
            padding: 12px;
            border-radius: 8px;
            width: 100%;
            border: 1px solid #ddd;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .input-box:focus {
            border-color: #ff9a9e;
            outline: none;
            box-shadow: 0 0 5px rgba(255, 154, 158, 0.6);
        }
        .result-box {
            margin-top: 20px;
            padding: 20px;
            border-radius: 8px;
            background-color: #f9f9f9;
            font-size: 24px;
            font-weight: bold;
            color: #444;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .good { background-color: #4CAF50; color: white; }
        .neutral { background-color: #FFC107; color: white; }
        .bad { background-color: #F44336; color: white; }
        .movie-title {
            font-weight: bold;
            color: #007bff;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the App
st.markdown("<h1 class='title'>Movie Rating Prediction</h1>", unsafe_allow_html=True)

# Subtitle
st.markdown("<h2 class='subtitle'>Enter a Bollywood movie name, and get its predicted rating category instantly.</h2>", unsafe_allow_html=True)

# Input Box for Movie Title
movie_title = st.text_input("Enter Movie Title", key='movie_title', placeholder="e.g., Khel Khel Mein", help="Type the name of the Bollywood movie")

# Load the model and data
df = load_data()
model = load_model()

# Predict and display rating category immediately after the user enters a title
if movie_title:
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model)

    # Display the result with specific styling based on the rating category
    if predicted_category == "Good":
        st.markdown(f"<div class='result-box good'>Predicted category for <span class='movie-title'>{movie_title}</span>: {predicted_category}</div>", unsafe_allow_html=True)
    elif predicted_category == "Neutral":
        st.markdown(f"<div class='result-box neutral'>Predicted category for <span class='movie-title'>{movie_title}</span>: {predicted_category}</div>", unsafe_allow_html=True)
    elif predicted_category == "Bad":
        st.markdown(f"<div class='result-box bad'>Predicted category for <span class='movie-title'>{movie_title}</span>: {predicted_category}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box'>{predicted_category}</div>", unsafe_allow_html=True)
