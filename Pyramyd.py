# Load necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import re, string
from nltk.corpus import stopwords
import nltk
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT 
import uvicorn
import os
import torch

# Preprocess setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model)

app = FastAPI()

# Loading dataset
raw_df = pd.read_csv("G2 software - CRM Category Product Overviews.csv")

# Clean 'description' column
def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(r'[^\x00-\x7f]', '', text).lower()
    text = re.sub(r'http\S+|@\S+', '', text)
    tokens = text.split()
    tokens = [t.translate(str.maketrans('', '', string.punctuation)) for t in tokens]
    return ' '.join([t for t in tokens if t not in stop_words])

# Clean and flatten 'Features' column
def flatten_features(text):
    if pd.isna(text): return ""
    categories = re.findall(r"Category['\"]?:\s*['\"](.*?)['\"]", text)
    description = re.findall(r"description['\"]?:\s*['\"](.*?)['\"]", text)
    return clean_text(' '.join(categories + description))


# Extract keywords from product features and description
def extract_keywords(text):
    if not text: return []
    keywords = kw_model.extract_keywords(text, stop_words='english', top_n=20)
    return [kw for kw, _ in keywords]

# Calculate Jaccard similarity for token-level comparison between user query and predicted keywords
def jaccard_similarity(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b)) if a and b else 0

# Clean and load semantic input and predicted keywords
raw_df['Cleaned_Features'] = raw_df['Features'].apply(flatten_features)
raw_df['Cleaned_Description'] = raw_df['description'].apply(clean_text)
raw_df['semantic_input'] = raw_df['Cleaned_Features'] + ' ' + raw_df['Cleaned_Description']
raw_df['predicted_keywords'] = raw_df['semantic_input'].apply(extract_keywords)

# Normalize ratings for final scoring
min_rating = raw_df['rating'].min()
max_rating = raw_df['rating'].max()
raw_df['normalized_rating'] = (raw_df['rating'] - min_rating) / (max_rating - min_rating)

# API endpoint to handle vendor qualification queries
class VendorQuery(BaseModel):
    software_category: str
    capabilities: List[str]

# Endpoint to handle vendor qualification queries
@app.post("/vendor_qualification")
def vendor_qualification(query: VendorQuery):
    df = raw_df[raw_df['main_category'].str.lower() == query.software_category.lower()].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail="No vendors found for given category.")
    user_query = ' '.join(query.capabilities).lower()
    user_tokens = user_query.split()
    corpus = df['semantic_input'].tolist()
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cos_scores_tensor = util.cos_sim(query_embedding, corpus_embeddings).squeeze()
    cos_scores = cos_scores_tensor.cpu().tolist()
    df['semantic_similarity'] = cos_scores
    df['token_overlap'] = df['predicted_keywords'].apply(lambda x: jaccard_similarity(user_tokens, x))
    df['final_score'] = (
        0.6 * df['semantic_similarity'] +
        0.1 * df['token_overlap'] +
        0.3 * df['normalized_rating']
    )
    top_vendors = df[df['final_score'] > 0.4].sort_values(by='final_score', ascending=False).head(10)
    if top_vendors.empty:
        print("No vendors passed the scoring threshold.")
    return top_vendors[['product_name', 'final_score']].to_dict(orient='records')

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run("Pyramyd:app", host="127.0.0.1", port=8000, reload=True)
