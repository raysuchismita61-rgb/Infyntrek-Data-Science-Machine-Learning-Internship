import streamlit as st
import pandas as pd
import joblib
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


st.set_page_config(
    page_title="AI Task Management System",
    page_icon="🤖",
    layout="wide"
)


# Load data

df = pd.read_csv(
    "data/cleaned_tasks.csv"
)


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


# Title

st.title(
    "🤖 AI-Powered Task Management System"
)

st.write(
    "Automatically classify, prioritize and analyze tasks using Machine Learning and NLP."
)


# Sidebar

st.sidebar.header(
    "Enter Task Details"
)


description = st.sidebar.text_area(
    "Task Description",
    "Fix login authentication bug"
)

days = st.sidebar.number_input(
    "Days Remaining",
    min_value=1,
    max_value=365,
    value=5
)

hours = st.sidebar.number_input(
    "Estimated Hours",
    min_value=1,
    max_value=100,
    value=5
)

workload = st.sidebar.slider(
    "Current Workload",
    0,
    100,
    50
)


if st.sidebar.button(
    "Predict Task"
):

    cleaned = clean_text(
        description
    )


    # Classification

    text_features = vectorizer.transform(
        [cleaned]
    )

    category = classifier.predict(
        text_features
    )[0]


    # Priority

    category_encoded = category_encoder.transform(
        [category]
    )[0]


    features = [[
        days,
        hours,
        workload,
        category_encoded
    ]]


    priority = priority_model.predict(
        features
    )[0]


    st.subheader(
        "Prediction Result"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Task Category",
            category
        )


    with col2:

        st.metric(
            "Priority",
            priority
        )


    with col3:

        st.metric(
            "Current Workload",
            f"{workload}%"
        )


st.divider()


# Dataset statistics

st.subheader(
    "Dataset Overview"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Tasks",
        len(df)
    )


with col2:

    st.metric(
        "Categories",
        df["Category"].nunique()
    )


with col3:

    st.metric(
        "High Priority Tasks",
        len(
            df[
                df["Priority"].isin(
                    ["High", "Critical"]
                )
            ]
        )
    )


with col4:

    st.metric(
        "Users",
        df["Assigned_User"].nunique()
    )


st.subheader(
    "Task Category Distribution"
)

st.bar_chart(
    df["Category"].value_counts()
)


st.subheader(
    "Priority Distribution"
)

st.bar_chart(
    df["Priority"].value_counts()
)


st.subheader(
    "Current Workload"
)

workload_df = (
    df.groupby("Assigned_User")
    ["Current_Workload"]
    .mean()
)

st.bar_chart(
    workload_df
)