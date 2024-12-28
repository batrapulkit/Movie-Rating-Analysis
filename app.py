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

# Ensure the required NLTK resources are downloaded
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Check if the required NLTK data is available
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    st.error("Required NLTK resources are missing. Please check your environment.")
    raise

# Load the dataset
@st.cache_data  # Cache the dataset to prevent reloading every time
def load_data():
    df = pd.read_csv('movies.csv')  # Or use st.file_uploader() to upload the CSV file
    return df

df = load_data()

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
    tokens = word_tokenize(str(text).lower())
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Apply text preprocessing to the 'title' column
df['processed_text'] = df['title'].apply(preprocess_text)

# Replace zero ratings with a random value
def replace_zero_ratings_with_random_value(rating):
    if rating == 0:
        return random.uniform(5.1, 7)
    else:
        return rating

df['rating'] = df['rating'].apply(replace_zero_ratings_with_random_value)

# Drop duplicates based on title
df = df.drop_duplicates(subset='title', keep='last')

# Feature extraction: TF-IDF for movie title and adding 'ratings' as a feature
vectorizer = TfidfVectorizer(max_features=1000)
X_title = vectorizer.fit_transform(df['processed_text'])

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

# Function to predict rating category for a movie using the title
def predict_rating_category_from_dataset(title, df):
    # Check if the movie title exists in the dataset (case insensitive partial matching)
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        rating = pd.to_numeric(movie_data.iloc[0]['rating'], errors='coerce')  # Convert to numeric
        
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

# Streamlit interface
st.title('Movie Rating Prediction')
st.write("Enter a movie title, and I'll predict its rating category based on the dataset!")

movie_title = st.text_input("Enter Movie Title")

if movie_title:
    st.write("Predicting...")
    predicted_category = predict_rating_category_from_dataset(movie_title, df)
    st.write(f"Predicted Rating Category for '{movie_title}': {predicted_category}")
