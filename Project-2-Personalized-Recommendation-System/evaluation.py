import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "merged_data.csv")

data = pd.read_csv(DATA_PATH)


def precision_at_k(recommended_movies, relevant_movies, k=10):

    recommended_movies = recommended_movies[:k]

    if len(recommended_movies) == 0:
        return 0

    hits = len(
        set(recommended_movies)
        & set(relevant_movies)
    )

    return hits / k


# Example
user_id = data["userId"].iloc[0]

user_data = data[data["userId"] == user_id]

relevant_movies = user_data[
    user_data["rating"] >= 4
]["movieId"].tolist()

# Example recommendation list
recommended_movies = user_data.sort_values(
    "rating",
    ascending=False
)["movieId"].tolist()

score = precision_at_k(
    recommended_movies,
    relevant_movies,
    k=10
)

print("User ID:", user_id)
print("Precision@10:", score)