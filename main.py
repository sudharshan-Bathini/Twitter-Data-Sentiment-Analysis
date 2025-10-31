from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import nltk

# Initialize FastAPI
app = FastAPI(title="Twitter Sentiment Analysis API", version="1.0")

# Load model and features
with open("sentiment_analysis_model.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("word_features.pkl", "rb") as f:
    w_features = pickle.load(f)

# Function to extract features from input text
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# Input schema
class Tweet(BaseModel):
    text: str

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to the Twitter Sentiment Analysis API!"}

# Prediction route
@app.post("/predict")
def predict_sentiment(tweet: Tweet):
    text = tweet.text
    words = [w.lower() for w in text.split() if len(w) >= 3]
    result = classifier.classify(extract_features(words))
    return {"text": text, "sentiment": result}
