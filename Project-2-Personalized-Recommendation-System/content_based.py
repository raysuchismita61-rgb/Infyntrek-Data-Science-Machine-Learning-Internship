import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies.csv")

movies = pd.read_csv(DATA_PATH)

movies["genres"] = movies["genres"].fillna("")

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["genres"]
)

print("TF-IDF matrix shape:")
print(tfidf_matrix.shape)

# Similarity
similarity = cosine_similarity(tfidf_matrix)


def content_recommendations(movie_title, n=10):

    matches = movies[
        movies["title"].str.lower()
        == movie_title.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    index = matches.index[0]

    scores = list(
        enumerate(similarity[index])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:n+1]

    movie_indices = [
        x[0] for x in scores
    ]

    result = movies.iloc[
        movie_indices
    ][["movieId", "title", "genres"]].copy()

    result["similarity"] = [
        x[1] for x in scores
    ]

    return result


# Example
movie = movies["title"].iloc[0]

print("\nRecommendations for:", movie)

print(
    content_recommendations(movie)
)