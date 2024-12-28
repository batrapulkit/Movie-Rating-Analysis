import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from scipy.sparse import hstack

# Ensure that the required NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('movies.csv')

def categorize_rating(rating):
    if rating > 7:
        return 'Good'
    elif 4.5 <= rating <= 6.9:
        return 'Neutral'
    else:
        return 'Bad'

# Apply the function to the 'rating' column to create a new 'rating_category' column
df['rating_category'] = df['rating'].apply(categorize_rating)

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

# Apply text preprocessing to the 'title' column (adjust if needed for another column)
df['processed_text'] = df['title'].apply(preprocess_text)

# Replace zero ratings with a random value
def replace_zero_ratings_with_random_value(rating):
    if rating == 0:
        return random.uniform(5.1, 7)  # Random value between 5.1 and 7
    else:
        return rating

# Apply the function to the 'rating' column
df['rating'] = df['rating'].apply(replace_zero_ratings_with_random_value)

# Drop duplicates based on title
df = df.drop_duplicates(subset='title', keep='last')

# Feature extraction: TF-IDF for the movie title and adding the 'ratings' as a feature
vectorizer = TfidfVectorizer(max_features=1000)
X_title = vectorizer.fit_transform(df['processed_text'])

# Adding 'ratings' as a numerical feature to the dataset
X_ratings = np.array(df['rating']).reshape(-1, 1)

# Combine TF-IDF features and ratings (concatenating the matrices)
X = hstack([X_title, X_ratings])

# Labels (rating category)
y = df['rating_category']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print classification report (optional)
print(classification_report(y_test, y_pred))

# Function to predict the rating category for a movie using the movie title
def predict_rating_category_from_dataset(title, df):
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
st.title('Movie Rating Prediction')
st.write("Enter a movie title, and I'll predict its rating category based on the dataset!")

movie_title = st.text_input("Enter Movie Title")

if movie_title:
    predicted_category = predict_rating_category_from_dataset(movie_title, df)
    st.write(f"Predicted Rating Category for '{movie_title}': {predicted_category}")
