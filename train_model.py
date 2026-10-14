import os
import csv
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
DATASET_PATH = "spam.csv"
MODEL_PATH = os.path.join("models", "spam_model.joblib")
def load_dataset(path):
    """Load either a regular CSV or the original tab-separated dataset."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Place spam.csv in the project folder."
        )
    # First, try a regular CSV with a header.
    try:
        df = pd.read_csv(path, encoding="latin-1")
        df.columns = [
            str(column).strip().lower() for column in df.columns
        ]
        if {"label", "message"}.issubset(df.columns):
            df = df[["label", "message"]]
        elif {"v1", "v2"}.issubset(df.columns):
            df = df.rename(
                columns={"v1": "label", "v2": "message"}
            )[["label", "message"]]
        else:
            # Try the original UCI tab-separated format.
            raise ValueError("Trying tab-separated format")
    except (pd.errors.ParserError, ValueError):
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "message"],
            quoting=csv.QUOTE_NONE,
            encoding="latin-1",
            on_bad_lines="error",
        )
    df = df.dropna().copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["spam", "ham"])]
    df = df[df["message"] != ""]
    if df.empty or df["label"].nunique() < 2:
        raise ValueError(
            "No valid dataset found. Check that labels are spam/ham "
            "and messages are in the correct column."
        )
    return df
def main():
    try:
        df = load_dataset(DATASET_PATH)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print("Could not load dataset:")
        print(error)
        return
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english"
            ),
        ),
        ("classifier", MultinomialNB()),
    ])
    print("Training spam filtering model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\n=== Model Evaluation ===")
    print(f"Total messages: {len(df)}")
    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, predictions, zero_division=0
        )
    )
    print("Confusion Matrix (rows=true, columns=predicted)")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved successfully to: {MODEL_PATH}")
if __name__ == "__main__":
    main()
