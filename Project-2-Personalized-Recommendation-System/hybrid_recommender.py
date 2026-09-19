import pandas as pd
import numpy as np
import os

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


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

print("Loading data...")

ratings = pd.read_csv(
    RATINGS_PATH,
    usecols=["userId", "movieId", "rating"]
)

movies = pd.read_csv(
    MOVIES_PATH,
    usecols=["movieId", "title", "genres"]
)

movies["genres"] = movies["genres"].fillna("")

print("Ratings Shape:", ratings.shape)
print("Movies Shape:", movies.shape)


# ==========================================
# CONTENT-BASED MODEL
# ==========================================

print("\nCreating TF-IDF content model...")

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["genres"]
)

print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ==========================================
# CREATE MOVIE ID MAPPING
# ==========================================

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(
        movies["movieId"]
    )
}

index_to_movie = {
    index: movie_id
    for movie_id, index in movie_to_index.items()
}


# ==========================================
# CREATE USER / MOVIE MAPPINGS
# ==========================================

print("\nCreating ID mappings...")

user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

rating_movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}


# ==========================================
# CREATE SPARSE USER-ITEM MATRIX
# ==========================================

print("\nCreating sparse User-Item matrix...")

row_indices = ratings["userId"].map(
    user_to_index
).values

column_indices = ratings["movieId"].map(
    rating_movie_to_index
).values

values = ratings["rating"].values.astype(
    np.float32
)

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
    "Sparse User-Item matrix:",
    user_item_matrix.shape
)

print(
    "Number of ratings:",
    user_item_matrix.nnz
)


# ==========================================
# ITEM-BASED COLLABORATIVE FILTERING
# ==========================================

print(
    "\nTraining Item-Based Collaborative Filtering..."
)

item_user_matrix = user_item_matrix.T.tocsr()

item_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11,
    n_jobs=-1
)

item_model.fit(item_user_matrix)

print(
    "Item-Based Collaborative Filtering trained."
)


# ==========================================
# POPULARITY SCORE
# ==========================================

print("\nCalculating movie popularity...")

movie_rating_stats = ratings.groupby(
    "movieId"
)["rating"].agg(
    ["mean", "count"]
)

# Weighted popularity
C = ratings["rating"].mean()

m = movie_rating_stats["count"].quantile(0.90)

movie_rating_stats["weighted_score"] = (
    (
        movie_rating_stats["count"]
        /
        (
            movie_rating_stats["count"] + m
        )
    )
    *
    movie_rating_stats["mean"]
    +
    (
        m
        /
        (
            movie_rating_stats["count"] + m
        )
    )
    *
    C
)

# Normalize
max_popularity = (
    movie_rating_stats["weighted_score"].max()
)

if max_popularity != 0:

    movie_rating_stats["popularity"] = (
        movie_rating_stats["weighted_score"]
        /
        max_popularity
    )

else:

    movie_rating_stats["popularity"] = 0


# ==========================================
# HYBRID RECOMMENDER
# ==========================================

def hybrid_recommendations(
    user_id,
    n=10,
    alpha=0.7
):

    """
    Hybrid recommendation system.

    alpha:
        Weight for collaborative filtering.

    1-alpha:
        Weight for content-based filtering.
    """

    if user_id not in user_to_index:

        print(
            "User ID not found:",
            user_id
        )

        return pd.DataFrame()


    # ======================================
    # USER RATINGS
    # ======================================

    user_index = user_to_index[user_id]

    user_vector = user_item_matrix[
        user_index
    ]

    rated_indices = user_vector.indices

    rated_scores = user_vector.data


    if len(rated_indices) == 0:

        print(
            "User has no ratings."
        )

        return movies.head(n)


    # ======================================
    # GET WATCHED MOVIES
    # ======================================

    watched_movie_ids = [
        movie_ids[index]
        for index in rated_indices
    ]

    watched_set = set(
        watched_movie_ids
    )


    # ======================================
    # CONTENT-BASED SCORE
    # ======================================

    print(
        "\nCalculating content-based scores..."
    )

    # Use only highly-rated movies
    rated_pairs = list(
        zip(
            rated_indices,
            rated_scores
        )
    )

    rated_pairs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Limit to avoid unnecessary computation
    rated_pairs = rated_pairs[:20]


    # Create user content profile
    user_profile = None

    for movie_index, rating in rated_pairs:

        movie_id = movie_ids[
            movie_index
        ]

        content_index = movie_to_index.get(
            movie_id
        )

        if content_index is None:
            continue

        movie_vector = (
            tfidf_matrix[content_index]
            * float(rating)
        )

        if user_profile is None:

            user_profile = movie_vector

        else:

            user_profile = (
                user_profile
                + movie_vector
            )


    if user_profile is None:

        content_scores = np.zeros(
            len(movies)
        )

    else:

        # Sparse matrix multiplication
        content_scores = (
            user_profile
            @
            tfidf_matrix.T
        ).toarray().ravel()

        max_content = content_scores.max()

        if max_content != 0:

            content_scores = (
                content_scores
                /
                max_content
            )


    # ======================================
    # COLLABORATIVE FILTERING SCORE
    # ======================================

    print(
        "Calculating collaborative scores..."
    )

    collaborative_scores = {}

    # Use top-rated movies only
    for movie_index, rating in rated_pairs:

        original_movie_id = movie_ids[
            movie_index
        ]

        movie_vector = item_user_matrix[
            movie_index
        ]

        distances, indices = (
            item_model.kneighbors(
                movie_vector,
                n_neighbors=11
            )
        )

        for neighbor_index, distance in zip(
            indices[0][1:],
            distances[0][1:]
        ):

            similarity = 1 - distance

            recommended_movie_id = (
                movie_ids[neighbor_index]
            )

            if recommended_movie_id in watched_set:
                continue

            score = (
                similarity
                * float(rating)
            )

            collaborative_scores[
                recommended_movie_id
            ] = (
                collaborative_scores.get(
                    recommended_movie_id,
                    0
                )
                + score
            )


    # ======================================
    # CREATE FINAL SCORES
    # ======================================

    print(
        "Combining recommendation scores..."
    )

    results = movies.copy()

    results["content_score"] = (
        content_scores
    )

    results["collaborative_score"] = (
        results["movieId"].map(
            collaborative_scores
        ).fillna(0)
    )


    # Normalize collaborative scores

    max_cf = results[
        "collaborative_score"
    ].max()

    if max_cf != 0:

        results["collaborative_score"] = (
            results["collaborative_score"]
            /
            max_cf
        )


    # ======================================
    # HYBRID SCORE
    # ======================================

    results["hybrid_score"] = (
        alpha
        *
        results["collaborative_score"]
        +
        (1 - alpha)
        *
        results["content_score"]
    )


    # ======================================
    # REMOVE WATCHED MOVIES
    # ======================================

    results = results[
        ~results["movieId"].isin(
            watched_set
        )
    ]


    # ======================================
    # SORT
    # ======================================

    results = results.sort_values(
        "hybrid_score",
        ascending=False
    )


    # ======================================
    # RETURN TOP N
    # ======================================

    return results[
        [
            "movieId",
            "title",
            "genres",
            "hybrid_score"
        ]
    ].head(n)


# ==========================================
# TEST
# ==========================================

test_user_id = user_ids[0]

print(
    "\n======================================"
)

print(
    "HYBRID RECOMMENDATIONS"
)

print(
    "======================================"
)

recommendations = hybrid_recommendations(
    test_user_id,
    n=10,
    alpha=0.7
)

print(
    "\nTop Recommendations:"
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
    "Hybrid Recommendation System Completed!"
)

print(
    "======================================"
)