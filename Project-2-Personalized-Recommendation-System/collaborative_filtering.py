import pandas as pd
import numpy as np
import os

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RATINGS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ratings.csv"
)

MOVIES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "movies.csv"
)

print("Loading ratings...")

ratings = pd.read_csv(
    RATINGS_PATH,
    usecols=["userId", "movieId", "rating"]
)

movies = pd.read_csv(
    MOVIES_PATH
)

print("Ratings Shape:", ratings.shape)
print("Movies Shape:", movies.shape)
print("\nCreating ID mappings...")

user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

row_indices = ratings["userId"].map(
    user_to_index
).values

column_indices = ratings["movieId"].map(
    movie_to_index
).values

values = ratings["rating"].values

print("\nCreating sparse User-Item matrix...")

user_item_matrix = csr_matrix(
    (
        values,
        (row_indices, column_indices)
    ),
    shape=(
        len(user_ids),
        len(movie_ids)
    ),
    dtype=np.float32
)

print(
    "Sparse matrix shape:",
    user_item_matrix.shape
)

print(
    "Number of ratings:",
    user_item_matrix.nnz
)

print(
    "\nTraining User-Based Collaborative Filtering..."
)

user_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11,
    n_jobs=-1
)

user_model.fit(user_item_matrix)

print(
    "User-Based model trained successfully."
)

def user_based_recommendations(
    user_id,
    n=10
):

    if user_id not in user_to_index:

        print(
            "User ID not found:",
            user_id
        )

        return pd.DataFrame()

    user_index = user_to_index[user_id]

    user_vector = user_item_matrix[
        user_index
    ]

    distances, indices = user_model.kneighbors(
        user_vector,
        n_neighbors=11
    )

    scores = {}

    # Similar users
    for similar_index, distance in zip(
        indices[0][1:],
        distances[0][1:]
    ):

        similarity = 1 - distance

        similar_user_id = user_ids[
            similar_index
        ]

        similar_ratings = ratings[
            ratings["userId"] == similar_user_id
        ]

        for _, row in similar_ratings.iterrows():

            movie_id = row["movieId"]
            rating = row["rating"]

            # Skip movies already rated
            user_rated = ratings[
                (ratings["userId"] == user_id)
                &
                (ratings["movieId"] == movie_id)
            ]

            if not user_rated.empty:
                continue

            scores[movie_id] = (
                scores.get(movie_id, 0)
                + similarity * rating
            )

    if not scores:
        return pd.DataFrame()

    recommendations = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n]

    result = pd.DataFrame(
        recommendations,
        columns=[
            "movieId",
            "score"
        ]
    )

    result = result.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId",
        how="left"
    )

    return result[
        [
            "movieId",
            "title",
            "genres",
            "score"
        ]
    ]

print(
    "\nTraining Item-Based Collaborative Filtering..."
)

item_user_matrix = user_item_matrix.T

item_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11,
    n_jobs=-1
)

item_model.fit(item_user_matrix)

print(
    "Item-Based model trained successfully."
)
def item_based_recommendations(
    user_id,
    n=10
):

    if user_id not in user_to_index:

        print(
            "User ID not found:",
            user_id
        )

        return pd.DataFrame()

    user_index = user_to_index[user_id]

    user_vector = user_item_matrix[
        user_index
    ]

    rated_movie_indices = (
        user_vector.indices
    )

    rated_movie_scores = (
        user_vector.data
    )

    scores = {}

    for movie_index, rating in zip(
        rated_movie_indices,
        rated_movie_scores
    ):

        movie_vector = item_user_matrix[
            movie_index
        ]

        distances, indices = item_model.kneighbors(
            movie_vector,
            n_neighbors=11
        )

        for neighbor_index, distance in zip(
            indices[0][1:],
            distances[0][1:]
        ):

            similarity = 1 - distance

            recommended_movie_id = movie_ids[
                neighbor_index
            ]

            original_movie_id = movie_ids[
                movie_index
            ]
            if ratings[
                (ratings["userId"] == user_id)
                &
                (ratings["movieId"] == recommended_movie_id)
            ].empty:

                scores[
                    recommended_movie_id
                ] = (
                    scores.get(
                        recommended_movie_id,
                        0
                    )
                    + similarity * rating
                )

    if not scores:
        return pd.DataFrame()

    recommendations = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n]

    result = pd.DataFrame(
        recommendations,
        columns=[
            "movieId",
            "score"
        ]
    )

    result = result.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId",
        how="left"
    )

    return result[
        [
            "movieId",
            "title",
            "genres",
            "score"
        ]
    ]

test_user_id = user_ids[0]

print(
    "\n======================================"
)

print(
    "USER-BASED RECOMMENDATIONS"
)

print(
    "======================================"
)

print(
    user_based_recommendations(
        test_user_id,
        n=10
    )
)


print(
    "\n======================================"
)

print(
    "ITEM-BASED RECOMMENDATIONS"
)

print(
    "======================================"
)

print(
    item_based_recommendations(
        test_user_id,
        n=10
    )
)


print(
    "\n======================================"
)

print(
    "Collaborative Filtering Completed!"
)

print(
    "======================================"
)