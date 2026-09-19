import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ratings_path = os.path.join(DATA_DIR, "ratings.csv")
movies_path = os.path.join(DATA_DIR, "movies.csv")

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

print("Ratings Shape:", ratings.shape)
print("Movies Shape:", movies.shape)

print("\nRatings:")
print(ratings.head())

print("\nMovies:")
print(movies.head())

print("\nMissing values in ratings:")
print(ratings.isnull().sum())

print("\nMissing values in movies:")
print(movies.isnull().sum())

# Remove duplicate records
ratings = ratings.drop_duplicates()
movies = movies.drop_duplicates()

# Remove rows with missing essential values
ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
movies = movies.dropna(subset=["movieId", "title"])

# Keep valid ratings
ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5)]

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

output_path = os.path.join(DATA_DIR, "merged_data.csv")
data.to_csv(output_path, index=False)

print("\nCleaned data saved to:", output_path)

print("\nFinal shape:", data.shape)
print("\nRating statistics:")
print(data["rating"].describe())