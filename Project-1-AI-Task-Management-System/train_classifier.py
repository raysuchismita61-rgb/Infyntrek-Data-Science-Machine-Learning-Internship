import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)


df = pd.read_csv("data/cleaned_tasks.csv")

X = df["Cleaned_Description"]

y = df["Category"]


# TF-IDF

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X)


# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Naive Bayes

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)


nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)

nb_precision = precision_score(
    y_test,
    nb_predictions,
    average="weighted"
)

nb_recall = recall_score(
    y_test,
    nb_predictions,
    average="weighted"
)


print("\n========== NAIVE BAYES ==========")

print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)

print(
    classification_report(
        y_test,
        nb_predictions
    )
)


# SVM

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)


svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

svm_precision = precision_score(
    y_test,
    svm_predictions,
    average="weighted"
)

svm_recall = recall_score(
    y_test,
    svm_predictions,
    average="weighted"
)


print("\n========== SVM ==========")

print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)

print(
    classification_report(
        y_test,
        svm_predictions
    )
)


# Save best model

os.makedirs("models", exist_ok=True)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

joblib.dump(
    svm_model,
    "models/task_classifier.pkl"
)

print("\nModel saved successfully!")