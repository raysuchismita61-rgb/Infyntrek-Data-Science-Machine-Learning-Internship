import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


df = pd.read_csv("data/tasks.csv")

df["Cleaned_Description"] = df["Task_Description"].apply(clean_text)

df.to_csv("data/cleaned_tasks.csv", index=False)

print("Text preprocessing completed!")

print(
    df[
        [
            "Task_Description",
            "Cleaned_Description"
        ]
    ].head(10)
)