import pandas as pd
import numpy as np
import os

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# ==========================================
# PATHS
# ==========================================

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


# ==========================================
# LOAD DATA
# ==========================================

print("Loading ratings...")

ratings = pd.read_csv(
    RATINGS_PATH,
    usecols=[
        "userId",
        "movieId",
        "rating"
    ]
)

movies = pd.read_csv(
    MOVIES_PATH,
    usecols=[
        "movieId",
        "title",
        "genres"
    ]
)

print(
    "Ratings Shape:",
    ratings.shape
)

print(
    "Movies Shape:",
    movies.shape
)


# ==========================================
# CREATE ID MAPPINGS
# ==========================================

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

print(
    "\nCreating sparse User-Item matrix..."
)

row_indices = ratings["userId"].map(
    user_to_index
).values

column_indices = ratings["movieId"].map(
    movie_to_index
).values

values = ratings["rating"].astype(
    np.float32
).values

user_item_matrix = csr_matrix(
    (
        values,
        (
            row_indices,
            column_indices
        )
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
    "\nTraining Truncated SVD..."
)

n_components = 20

svd = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

user_latent = svd.fit_transform(
    user_item_matrix
)

movie_latent = svd.components_

print(
    "SVD training completed."
)

print(
    "User latent matrix shape:",
    user_latent.shape
)

print(
    "Movie latent matrix shape:",
    movie_latent.shape
)

explained_variance = (
    svd.explained_variance_ratio_.sum()
)

print(
    "\nExplained variance:",
    round(
        explained_variance,
        4
    )
)
def recommend_movies(
    user_id,
    n=10
):

    if user_id not in user_to_index:

        print(
            "User ID not found:",
            user_id
        )

        return pd.DataFrame()


    user_index = user_to_index[
        user_id
    ]

    # User's latent representation
    user_vector = user_latent[
        user_index
    ]
    scores = (
        user_vector
        @
        movie_latent
    )
    rated_movies = ratings[
        ratings["userId"] == user_id
    ]["movieId"].values

    rated_set = set(
        rated_movies
    )
    recommendations = pd.DataFrame(
        {
            "movieId": movie_ids,
            "score": scores
        }
    )
    recommendations = recommendations[
        ~recommendations["movieId"].isin(
            rated_set
        )
    ]
    recommendations = recommendations.merge(
        movies,
        on="movieId",
        how="left"
    )
    recommendations = recommendations.sort_values(
        "score",
        ascending=False
    )


    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "score"
        ]
    ].head(n)

test_user_id = user_ids[0]

print(
    "\n======================================"
)

print(
    "SVD RECOMMENDATIONS"
)

print(
    "======================================"
)

recommendations = recommend_movies(
    test_user_id,
    n=10
)

print(
    recommendations.to_string(
        index=False
    )
)

print(
    "\n======================================"
)

print(
    "Matrix Factorization Completed!"
)

print(
    "======================================"
)