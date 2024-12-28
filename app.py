import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    return df

# Preprocessing text (used to be part of the training process)
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(str(text).lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Function to predict the rating category based on the movie title and rating
def predict_rating_category_from_dataset(title, model, df):
    # Check if the movie title exists in the dataset
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        # Extract the rating from the dataset
        rating = movie_data.iloc[0]['rating']
        
        # Preprocess the movie title
        processed_title = preprocess_text(movie_data.iloc[0]['title'])
        
        # Re-create the TF-IDF vectorizer (since we don't have a saved vectorizer)
        vectorizer = TfidfVectorizer(max_features=1000)
        X_title = vectorizer.fit_transform(df['title'].apply(preprocess_text))
        
        # Convert the processed title to a TF-IDF vector
        title_vector = vectorizer.transform([processed_title])
        
        # Prepare the features by combining the TF-IDF vector and the rating
        X_input = np.hstack([title_vector.toarray(), np.array([[rating]])])
        
        # Predict the rating category
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
