import pandas as pd
import os
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)


df = pd.read_csv("data/cleaned_tasks.csv")


# Encode category

category_encoder = LabelEncoder()

df["Category_Encoded"] = category_encoder.fit_transform(
    df["Category"]
)


# Features

X = df[
    [
        "Days_Remaining",
        "Estimated_Hours",
        "Current_Workload",
        "Category_Encoded"
    ]
]

y = df["Priority"]


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Random Forest

rf = RandomForestClassifier(
    random_state=42
)


# GridSearchCV

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}


grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


print("Training Random Forest with GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)


best_model = grid_search.best_estimator_

print("\nBest Parameters:")

print("\nBest Parameters:")
print(grid_search.best_params_)


# Prediction

predictions = best_model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)


print("\n========== PRIORITY MODEL ==========")

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)


print(
    classification_report(
        y_test,
        predictions
    )
)


# Save model

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/priority_model.pkl"
)

joblib.dump(
    category_encoder,
    "models/category_encoder.pkl"
)

print("\nPriority model saved successfully!")