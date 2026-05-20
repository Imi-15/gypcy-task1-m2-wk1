import string
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Download required NLTK resources on first run.
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

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
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


def lemmatize_text(text: str, lemmatizer: WordNetLemmatizer) -> str:
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Comment", "Sentiment"])
    df["Sentiment"] = df["Sentiment"].astype(int)
    df["Cleaned_Comment"] = df["Comment"].apply(clean_text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    df["Processed_Comment"] = df["Cleaned_Comment"].apply(
        lambda text: tokenize_and_remove_stopwords(text, stop_words)
    )
    df["Final_Processed_Comment"] = df["Processed_Comment"].apply(
        lambda text: lemmatize_text(text, lemmatizer)
    )
    return df


def train_sentiment_model(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["Final_Processed_Comment"])
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    processed = tokenize_and_remove_stopwords(cleaned, stop_words)
    final_text = lemmatize_text(processed, lemmatizer)
    features = vectorizer.transform([final_text])
    prediction = model.predict(features)[0]
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map.get(prediction, "Unknown")


def main():
    local_data = load_local_data()
    uploaded_file = st.file_uploader("Upload sentiment_data.csv", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Loaded uploaded dataset successfully.")
    elif local_data is not None:
        df = local_data
        st.info(f"Loaded local dataset from `{DATA_FILENAME}`.")
    else:
        st.warning(
            "No local dataset found. Please upload `sentiment_data.csv` using the uploader above."
        )
        st.stop()

    if "Comment" not in df.columns or "Sentiment" not in df.columns:
        st.error("Dataset must contain the columns `Comment` and `Sentiment`.")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("### Sentiment Label Distribution")
    st.bar_chart(df["Sentiment"].value_counts().sort_index())

    if st.button("Preprocess and Train Model"):
        with st.spinner("Preprocessing data and training the model..."):
            df_processed = preprocess_dataframe(df)
            model, vectorizer, metrics = train_sentiment_model(df_processed)
            st.session_state["sentiment_model"] = model
            st.session_state["tfidf_vectorizer"] = vectorizer
            st.session_state["metrics"] = metrics
            st.session_state["df_processed"] = df_processed

    if "sentiment_model" in st.session_state:
        metrics = st.session_state["metrics"]
        st.success("Model trained successfully!")
        st.write(
            f"**Accuracy:** {metrics['accuracy']:.4f}\n"
            f"**Precision:** {metrics['precision']:.4f}\n"
            f"**Recall:** {metrics['recall']:.4f}\n"
            f"**F1-score:** {metrics['f1']:.4f}"
        )
        st.subheader("Classification Report")
        st.text(metrics["report"])

        st.subheader("Test a sentence")
        user_text = st.text_area("Enter a sentence to analyze", height=120)
        if st.button("Predict Sentiment"):
            if not user_text.strip():
                st.warning("Please enter text before predicting.")
            else:
                result = predict_sentiment(
                    user_text,
                    st.session_state["sentiment_model"],
                    st.session_state["tfidf_vectorizer"],
                )
                st.info(f"Predicted sentiment: **{result}**")

        if st.checkbox("Show preprocessed data preview"):
            st.write(st.session_state["df_processed"].head())

    else:
        st.info("Click the button above to train the sentiment model.")


if __name__ == "__main__":
    main()
