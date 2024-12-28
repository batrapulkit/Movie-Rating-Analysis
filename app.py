from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Load your dataset
df = pd.read_csv('movies.csv')

# Preprocess the rating column to categorize it
def categorize_rating(rating):
    if rating > 7:
        return 'Good'
    elif 4.5 <= rating <= 6.9:
        return 'Neutral'
    else:
        return 'Bad'

df['rating_category'] = df['rating'].apply(categorize_rating)

# Prepare the data with just the rating as the feature
X = df[['rating']]  # Only use rating as the feature
y = df['rating_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
with open('sentiment_model.pkl', 'wb') as file:
    pickle.dump(model, file)
