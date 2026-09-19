import streamlit as st
import pandas as pd
import numpy as np
import os

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
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

@st.cache_data
def load_data():

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

    movies["genres"] = movies[
        "genres"
    ].fillna("")

    return ratings, movies


ratings, movies = load_data()


# ==========================================
# CONTENT MODEL
# ==========================================

@st.cache_resource
def create_content_model(
    movie_genres
):

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        movie_genres
    )

    return tfidf_matrix


tfidf_matrix = create_content_model(
    movies["genres"]
)


# ==========================================
# CREATE SPARSE USER-ITEM MATRIX
# ==========================================

@st.cache_resource
def create_sparse_matrix(
    ratings
):

    user_ids = ratings[
        "userId"
    ].unique()

    movie_ids = ratings[
        "movieId"
    ].unique()

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(
            user_ids
        )
    }

    movie_to_index = {
        movie_id: index
        for index, movie_id in enumerate(
            movie_ids
        )
    }

    rows = ratings[
        "userId"
    ].map(
        user_to_index
    ).values

    columns = ratings[
        "movieId"
    ].map(
        movie_to_index
    ).values

    values = ratings[
        "rating"
    ].astype(
        np.float32
    ).values

    matrix = csr_matrix(
        (
            values,
            (rows, columns)
        ),
        shape=(
            len(user_ids),
            len(movie_ids)
        ),
        dtype=np.float32
    )

    return (
        matrix,
        user_ids,
        movie_ids,
        user_to_index,
        movie_to_index
    )


(
    user_item_matrix,
    user_ids,
    rating_movie_ids,
    user_to_index,
    rating_movie_to_index
) = create_sparse_matrix(
    ratings
)


# ==========================================
# ITEM-BASED COLLABORATIVE MODEL
# ==========================================

@st.cache_resource
def create_item_model(
    _user_item_matrix
):

    item_user_matrix = (
        _user_item_matrix.T.tocsr()
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=11,
        n_jobs=-1
    )

    model.fit(
        item_user_matrix
    )

    return (
        model,
        item_user_matrix
    )


item_model, item_user_matrix = (
    create_item_model(
        user_item_matrix
    )
)


# ==========================================
# MOVIE ID → CONTENT INDEX
# ==========================================

movie_content_index = {
    movie_id: index
    for index, movie_id in enumerate(
        movies["movieId"]
    )
}


# ==========================================
# POPULARITY
# ==========================================

@st.cache_data
def calculate_popularity(
    ratings
):

    stats = ratings.groupby(
        "movieId"
    )["rating"].agg(
        ["mean", "count"]
    )

    overall_mean = ratings[
        "rating"
    ].mean()

    minimum_votes = stats[
        "count"
    ].quantile(0.90)

    stats["weighted_score"] = (
        (
            stats["count"]
            /
            (
                stats["count"]
                + minimum_votes
            )
        )
        * stats["mean"]
        +
        (
            minimum_votes
            /
            (
                stats["count"]
                + minimum_votes
            )
        )
        * overall_mean
    )

    max_score = stats[
        "weighted_score"
    ].max()

    if max_score > 0:

        stats["popularity"] = (
            stats["weighted_score"]
            /
            max_score
        )

    else:

        stats["popularity"] = 0.0

    return stats


popularity = calculate_popularity(
    ratings
)


# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend(
    user_id,
    n=10,
    alpha=0.7
):

    # ======================================
    # CHECK USER
    # ======================================

    if user_id not in user_to_index:

        return pd.DataFrame()


    # ======================================
    # USER VECTOR
    # ======================================

    user_index = user_to_index[
        user_id
    ]

    user_vector = user_item_matrix[
        user_index
    ]

    rated_indices = (
        user_vector.indices
    )

    rated_scores = (
        user_vector.data
    )


    # ======================================
    # NO RATINGS
    # ======================================

    if len(rated_indices) == 0:

        popular_movies = (
            popularity
            .sort_values(
                "popularity",
                ascending=False
            )
            .head(n)
            .reset_index()
        )

        popular_movies = popular_movies.merge(
            movies,
            on="movieId",
            how="left"
        )

        return popular_movies[
            [
                "movieId",
                "title",
                "genres",
                "popularity"
            ]
        ].rename(
            columns={
                "popularity": "score"
            }
        )


    # ======================================
    # WATCHED MOVIES
    # ======================================

    watched_ids = {
        rating_movie_ids[index]
        for index in rated_indices
    }


    # ======================================
    # TOP RATED MOVIES
    # ======================================

    rated_movies = list(
        zip(
            rated_indices,
            rated_scores
        )
    )

    rated_movies.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Use only top 20 rated movies
    rated_movies = rated_movies[:20]


    # ======================================
    # CONTENT-BASED SCORE
    # ======================================

    user_profile = None

    for movie_index, rating in rated_movies:

        movie_id = rating_movie_ids[
            movie_index
        ]

        content_index = (
            movie_content_index.get(
                movie_id
            )
        )

        if content_index is None:
            continue

        movie_vector = (
            tfidf_matrix[
                content_index
            ]
            * float(rating)
        )

        if user_profile is None:

            user_profile = movie_vector

        else:

            user_profile = (
                user_profile
                + movie_vector
            )


    if user_profile is not None:

        content_scores = (
            user_profile
            @
            tfidf_matrix.T
        ).toarray().ravel()

    else:

        content_scores = np.zeros(
            len(movies),
            dtype=np.float32
        )


    # Normalize content scores

    max_content = (
        content_scores.max()
    )

    if max_content > 0:

        content_scores = (
            content_scores
            /
            max_content
        )


    # ======================================
    # COLLABORATIVE FILTERING SCORE
    # ======================================

    collaborative_scores = {}


    for movie_index, rating in rated_movies:

        movie_vector = (
            item_user_matrix[
                movie_index
            ]
        )

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

            recommended_id = (
                rating_movie_ids[
                    neighbor_index
                ]
            )

            # Skip watched movies

            if recommended_id in watched_ids:

                continue

            similarity = (
                1 - distance
            )

            score = (
                similarity
                * float(rating)
            )

            collaborative_scores[
                recommended_id
            ] = (
                collaborative_scores.get(
                    recommended_id,
                    0
                )
                + score
            )


    # ======================================
    # RESULT DATAFRAME
    # ======================================

    result = movies.copy()

    result["content_score"] = (
        content_scores
    )

    result["collaborative_score"] = (
        result["movieId"]
        .map(
            collaborative_scores
        )
        .fillna(0)
    )


    # ======================================
    # NORMALIZE CF SCORE
    # ======================================

    max_cf = result[
        "collaborative_score"
    ].max()

    if max_cf > 0:

        result[
            "collaborative_score"
        ] = (
            result[
                "collaborative_score"
            ]
            /
            max_cf
        )


    # ======================================
    # HYBRID SCORE
    # ======================================

    result["score"] = (
        alpha
        *
        result[
            "collaborative_score"
        ]
        +
        (1 - alpha)
        *
        result[
            "content_score"
        ]
    )


    # ======================================
    # REMOVE WATCHED MOVIES
    # ======================================

    result = result[
        ~result["movieId"].isin(
            watched_ids
        )
    ]


    # ======================================
    # SORT
    # ======================================

    result = result.sort_values(
        "score",
        ascending=False
    )


    # ======================================
    # TOP N
    # ======================================

    return result[
        [
            "movieId",
            "title",
            "genres",
            "score"
        ]
    ].head(n)


# ==========================================
# DASHBOARD
# ==========================================

st.title(
    "🎬 Personalized Movie Recommendation System"
)

st.write(
    "Machine Learning based Hybrid Recommendation Engine"
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "Recommendation Settings"
)

selected_user = st.sidebar.selectbox(
    "Select User",
    sorted(
        ratings["userId"].unique()
    )
)

number = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# ==========================================
# STATISTICS
# ==========================================

col1, col2, col3, col4 = (
    st.columns(4)
)

col1.metric(
    "Users",
    f"{ratings['userId'].nunique():,}"
)

col2.metric(
    "Movies",
    f"{movies['movieId'].nunique():,}"
)

col3.metric(
    "Ratings",
    f"{len(ratings):,}"
)

col4.metric(
    "Average Rating",
    f"{ratings['rating'].mean():.2f}"
)


st.divider()


# ==========================================
# RECOMMENDATIONS
# ==========================================

st.subheader(
    f"Recommended Movies for User {selected_user}"
)

with st.spinner(
    "Generating personalized recommendations..."
):

    recommendations = recommend(
        selected_user,
        number,
        alpha=0.7
    )


if recommendations.empty:

    st.warning(
        "No recommendations available."
    )

else:

    for _, row in (
        recommendations.iterrows()
    ):

        st.markdown(
            f"### 🎥 {row['title']}"
        )

        st.write(
            f"**Genre:** {row['genres']}"
        )

        st.write(
            f"**Recommendation Score:** "
            f"{row['score']:.3f}"
        )

        st.divider()