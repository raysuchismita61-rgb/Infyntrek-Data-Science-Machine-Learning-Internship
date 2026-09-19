import joblib
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Load models

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

classifier = joblib.load(
    "models/task_classifier.pkl"
)

priority_model = joblib.load(
    "models/priority_model.pkl"
)

category_encoder = joblib.load(
    "models/category_encoder.pkl"
)


stop_words = set(
    stopwords.words("english")
)

stemmer = PorterStemmer()


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


def predict_task(
    description,
    days_remaining,
    estimated_hours,
    workload
):

    # Clean text

    cleaned = clean_text(
        description
    )


    # Task classification

    text_features = vectorizer.transform(
        [cleaned]
    )

    category = classifier.predict(
        text_features
    )[0]


    # Encode category

    category_encoded = category_encoder.transform(
        [category]
    )[0]


    # Priority prediction

    priority_features = [[
        days_remaining,
        estimated_hours,
        workload,
        category_encoded
    ]]


    priority = priority_model.predict(
        priority_features
    )[0]


    print("\n==============================")

    print("AI TASK MANAGEMENT RESULT")

    print("==============================")

    print(
        "Task:",
        description
    )

    print(
        "Predicted Category:",
        category
    )

    print(
        "Predicted Priority:",
        priority
    )

    print(
        "Days Remaining:",
        days_remaining
    )

    print(
        "Estimated Hours:",
        estimated_hours
    )

    print(
        "Current Workload:",
        workload
    )


if __name__ == "__main__":

    description = input(
        "\nEnter task description: "
    )

    days = int(
        input(
            "Days remaining: "
        )
    )

    hours = int(
        input(
            "Estimated hours: "
        )
    )

    workload = int(
        input(
            "Current workload (0-100): "
        )
    )


    predict_task(
        description,
        days,
        hours,
        workload
    )