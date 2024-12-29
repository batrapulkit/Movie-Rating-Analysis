import streamlit as st
import joblib

# Load the pipeline (which includes the vectorizer and the model)
model = joblib.load("sentiment_model.pkl")

# Custom CSS for styling
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
        }
        .input-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .input-box {
            font-size: 18px;
            padding: 10px;
            border-radius: 4px;
            width: 100%;
            border: 1px solid #ddd;
        }
        .submit-btn {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .submit-btn:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            font-size: 24px;
            text-align: center;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the App
st.markdown("<h1 class='title'>Bollywood Movie Rating Predictor</h1>", unsafe_allow_html=True)

# Movie Name Input
movie_name = st.text_input("Enter Bollywood Movie Name", "")

# Function to make predictions using the model (without manual vectorization)
def predict_rating(movie_name):
    # Use the pipeline to predict directly on the movie name
    prediction = model.predict([movie_name])
    return prediction[0]

# Predict Rating Button
if st.button("Get Rating"):
    if movie_name:
        # Clean up the movie name input (remove leading/trailing spaces)
        movie_name = movie_name.strip()

        # Use the model to predict the movie rating
        try:
            rating = predict_rating(movie_name)
            st.markdown(f"<p class='result'>The predicted rating for <b>{movie_name}</b> is: {rating:.2f}/10</p>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<p class='result'>Error: {str(e)}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='result'>Please enter a movie name!</p>", unsafe_allow_html=True)
