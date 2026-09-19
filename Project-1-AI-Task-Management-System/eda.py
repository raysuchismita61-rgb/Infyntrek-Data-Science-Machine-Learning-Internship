import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/cleaned_tasks.csv")

os.makedirs("outputs", exist_ok=True)

print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTask Categories:")
print(df["Category"].value_counts())

print("\nPriority Distribution:")
print(df["Priority"].value_counts())

# Category distribution

plt.figure(figsize=(8, 5))

df["Category"].value_counts().plot(kind="bar")

plt.title("Task Category Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Tasks")

plt.tight_layout()

plt.savefig("outputs/category_distribution.png")

plt.show()


# Priority distribution

plt.figure(figsize=(8, 5))

df["Priority"].value_counts().plot(kind="bar")

plt.title("Task Priority Distribution")
plt.xlabel("Priority")
plt.ylabel("Number of Tasks")

plt.tight_layout()

plt.savefig("outputs/priority_distribution.png")

plt.show()


# Workload distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Current_Workload"],
    bins=20,
    kde=True
)

plt.title("Current Workload Distribution")

plt.tight_layout()

plt.savefig("outputs/workload_distribution.png")

plt.show()