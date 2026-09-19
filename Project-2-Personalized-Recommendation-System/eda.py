import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "merged_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)

# -----------------------------
# 1. Rating distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(data["rating"], bins=10)

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.savefig(os.path.join(OUTPUT_DIR, "rating_distribution.png"))
plt.show()


# -----------------------------
# 2. Most rated movies
# -----------------------------

movie_counts = (
    data.groupby("title")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

movie_counts.sort_values().plot(kind="barh")

plt.title("Top 10 Most Rated Movies")
plt.xlabel("Number of Ratings")

plt.savefig(os.path.join(OUTPUT_DIR, "most_rated_movies.png"))
plt.show()


# -----------------------------
# 3. Average movie ratings
# -----------------------------

avg_ratings = (
    data.groupby("title")["rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Movies by Average Rating:")
print(avg_ratings)


# -----------------------------
# 4. User activity
# -----------------------------

user_counts = (
    data.groupby("userId")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\nMost active users:")
print(user_counts)

print("\nEDA completed.")