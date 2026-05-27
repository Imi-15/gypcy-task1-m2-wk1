import re
import string
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Sentiment Analysis with Streamlit")
st.markdown(
    "Upload or load `sentiment_data.csv`, train a sentiment model, and test text predictions directly in your browser."
)

DATA_FILENAME = "sentiment_data.csv"

@st.cache_data(show_spinner=False)
def load_local_data():
    csv_path = Path(__file__).resolve().parent / DATA_FILENAME
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return text


def tokenize_and_remove_stopwords(text: str, stop_words: set[str]) -> str:
    tokens = re.findall(r"\b\w+\b", text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Comment", "Sentiment"])
    df["Sentiment"] = df["Sentiment"].astype(int)
    df["Cleaned_Comment"] = df["Comment"].apply(clean_text)
    stop_words = set(ENGLISH_STOP_WORDS)
    df["Final_Processed_Comment"] = df["Cleaned_Comment"].apply(
        lambda text: tokenize_and_remove_stopwords(text, stop_words)
    )
    return df


def train_sentiment_model(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["Final_Processed_Comment"])
    y = df["Sentiment"]
    
    # Only use stratify if there are 2+ unique classes
    stratify_param = y if len(y.unique()) >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_param,
    )
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }
    return model, vectorizer, metrics


def predict_sentiment(text: str, model, vectorizer) -> str:
    cleaned = clean_text(text)
    stop_words = set(ENGLISH_STOP_WORDS)
    final_text = tokenize_and_remove_stopwords(cleaned, stop_words)
    features = vectorizer.transform([final_text])
    prediction = model.predict(features)[0]
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map.get(prediction, "Unknown")


def main():
    local_data = load_local_data()
    if local_data is None:
        st.error(
            "No local dataset found. Please place `sentiment_data.csv` in the same folder as this app."
        )
        st.stop()

    if "Comment" not in local_data.columns or "Sentiment" not in local_data.columns:
        st.error("Dataset must contain the columns `Comment` and `Sentiment`.")
        st.stop()

    try:
        df_processed = preprocess_dataframe(local_data)
        model, vectorizer, _ = train_sentiment_model(df_processed)
    except ValueError as e:
        st.error(f"Error training model: {str(e)}")
        st.info("Please ensure your sentiment_data.csv has valid sentiment values (0, 1, or 2).")
        st.stop()

    st.subheader("Check the sentiment of a single sentence")
    user_text = st.text_area("Enter a sentence to analyze", height=120)
    if st.button("Predict Sentiment"):
        if not user_text.strip():
            st.warning("Please enter text before predicting.")
        else:
            result = predict_sentiment(user_text, model, vectorizer)
            st.info(f"Predicted sentiment: **{result}**")

    st.caption("Note: The app uses a trained logistic regression model behind the scenes.")


if __name__ == "__main__":
    main()
