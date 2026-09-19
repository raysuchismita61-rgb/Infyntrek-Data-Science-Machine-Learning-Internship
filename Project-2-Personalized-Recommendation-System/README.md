# Personalized Recommendation System

## Project Overview

The Personalized Recommendation System is a Machine Learning project developed as part of my **Infyntrek Data Science & Machine Learning Internship**.

The system analyzes user ratings and movie information to generate personalized movie recommendations. It combines collaborative filtering and content-based filtering techniques to recommend movies based on user preferences and movie characteristics.

## Objectives

- Analyze user-movie rating data.
- Build a personalized movie recommendation system.
- Implement user-based collaborative filtering.
- Implement item-based collaborative filtering.
- Use content-based filtering based on movie genres.
- Develop a hybrid recommendation approach.
- Provide personalized movie recommendations through an interactive dashboard.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Natural Language Processing
- TF-IDF
- Collaborative Filtering
- Content-Based Filtering
- Streamlit

## Dataset

The project uses the **MovieLens dataset**, which contains movie information and user ratings.

### Dataset Files

- `ratings.csv` — User ratings for movies.
- `movies.csv` — Movie IDs, titles, and genres.

##  Recommendation Techniques

### 1. User-Based Collaborative Filtering

This approach identifies users with similar rating patterns and recommends movies based on the preferences of similar users.

### 2. Item-Based Collaborative Filtering

This approach identifies movies that are similar based on user rating patterns and recommends movies related to movies the user has already rated.

### 3. Content-Based Filtering

Movie genres are converted into numerical features using **TF-IDF** and used to identify movies with similar characteristics.

### 4. Hybrid Recommendation

The system combines collaborative filtering and content-based recommendations to produce personalized results.

## Project Workflow

1. Load the MovieLens dataset.
2. Perform data preprocessing.
3. Create the user-item rating matrix.
4. Calculate user and item similarities.
5. Build collaborative filtering models.
6. Extract movie genre features.
7. Apply TF-IDF vectorization.
8. Generate content-based recommendations.
9. Combine recommendation techniques.
10. Display recommendations through the Streamlit dashboard.

## Project Structure

```text
Project-2-Personalized-Recommendation-System/
│
├── data/
│   ├── ratings.csv
│   └── movies.csv
│
├── src/
│   ├── collaborative_filtering.py
│   ├── content_based.py
│   └── hybrid_recommendation.py
│
├── dashboard.py
└── README.md
