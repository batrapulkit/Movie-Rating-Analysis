import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model and dataset
@st.cache_resource
def load_model():
    """Load the pre-trained model."""
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_data
def load_data():
    """Load the dataset."""
    # Replace with the correct path to your dataset
    df = pd.read_csv('movies.csv')
    return df

# Function to categorize the rating into categories
def categorize_rating(rating):
    """Categorizes the movie rating."""
    if rating >= 7:
        return "Good"
    elif 5 <= rating < 7:
        return "Neutral"
    else:
        return "Bad"

# Function to predict the rating category for a movie using the movie title
def predict_rating_category_from_dataset(title, df):
    """Predicts the movie's rating category based on the title."""
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        # Extract the rating from the dataset and convert it to a numeric value
        rating = pd.to_numeric(movie_data.iloc[0]['rating'], errors='coerce')
        
        if not pd.isna(rating):
            # Use categorize_rating to classify the rating
            return categorize_rating(rating)
        else:
            return "Invalid rating data"
    else:
        return "Movie not found in dataset"

# Streamlit interface
st.title("Movie Rating Prediction")

movie_title = st.text_input("Enter Movie Title")

# When the user inputs a movie title
if movie_title:
    # Ensure that both the model and the data are loaded
    model = load_model()
    df = load_data()

    # Get the predicted category from the function
    predicted_category = predict_rating_category_from_dataset(movie_title, df)
    
    # Display the predicted category
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")
