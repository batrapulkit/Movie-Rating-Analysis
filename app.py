import streamlit as st
import pickle
import random
import numpy as np

# Load the trained model
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit interface
st.title('Movie Rating Prediction')
st.write("Enter the movie title and the rating to predict its rating category (Good, Neutral, or Bad).")

# Text input for the movie title and rating
movie_title = st.text_input("Enter Movie Title")
movie_rating = st.number_input("Enter Movie Rating", min_value=0.0, max_value=10.0, step=0.1)

if movie_title:
    # If the rating is 0, replace it with a random rating for demonstration purposes
    if movie_rating == 0:
        movie_rating = random.uniform(5.1, 7)  # Assign a random value between 5.1 and 7
    
    # Create a numpy array for the rating (this should be the same format as during training)
    X_input = np.array([[movie_rating]])

    # Predict the rating category
    prediction = model.predict(X_input)
    rating_category = prediction[0]
    
    # Display the prediction
    st.write(f"The predicted rating category for '{movie_title}' with a rating of {movie_rating} is: **{rating_category}**")
