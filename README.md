# Movie Rating Model

This project is a machine learning-based system designed to predict the rating category (Good, Neutral, Bad) of a movie based on its title and rating. The model uses logistic regression along with Natural Language Processing (NLP) techniques such as TF-IDF vectorization to process movie titles and ratings for prediction.

## Project Overview

The model is trained on a dataset containing movie titles and their ratings. It categorizes each movie's rating into three categories:

- **Good**: Rating > 7
- **Neutral**: Rating between 4.5 and 6.9
- **Bad**: Rating < 4.5

### Steps involved in the model:

1. **Data Preprocessing**:
   - Movies with zero ratings are replaced with random values between 5.1 and 7.
   - Movie titles are preprocessed using tokenization, stopword removal, and lemmatization.
   - Duplicates in movie titles are removed.

2. **Feature Extraction**:
   - The model uses TF-IDF vectorization to convert the movie titles into numerical features.
   - The movie ratings are used as an additional feature.
   - The features (TF-IDF + ratings) are combined for training.

3. **Model Training**:
   - Logistic Regression is used to classify the movie into a rating category.

4. **Model Evaluation**:
   - The model's performance is evaluated using a classification report, which includes metrics like precision, recall, and F1-score.

5. **Prediction**:
   - The model can predict the rating category for a given movie based on its title.

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `sklearn`
  - `nltk`
  - `numpy`
  - `scipy`

Install required libraries using:

```bash
pip install pandas scikit-learn nltk numpy scipy
