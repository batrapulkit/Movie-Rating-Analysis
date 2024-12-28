import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model():
    # Load the pre-trained model
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    return df

# Function to predict the rating category for a movie based on the rating
def predict_rating_category_from_dataset(title, model, df):
    # Check if the movie title exists in the dataset
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        # Extract the rating from the dataset and convert it to a numeric value
        rating = movie_data.iloc[0]['rating']
        
        # Prepare the features by adding the rating value
        X_input = np.array([[rating]])  # Only using the rating value as input
        
        # Predict the rating category using the pre-trained model
        predicted_category = model.predict(X_input)
        return predicted_category[0]
    else:
        return "Movie not found in dataset"

# Streamlit interface
st.title("Movie Rating Prediction")

# Load the model and dataset
model = load_model()
df = load_data()

movie_title = st.text_input("Enter Movie Title")

if movie_title:
    predicted_category = predict_rating_category_from_dataset(movie_title, model, df)
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")
