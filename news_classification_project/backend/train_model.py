# ...existing code...
import os
import sys
import zipfile
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def ensure_data_extracted(data_dir="data", zip_name="test.csv.zip", csv_name="test.csv"):
    zip_path = os.path.join(data_dir, zip_name)
    csv_path = os.path.join(data_dir, csv_name)
    if os.path.exists(zip_path) and not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
    return csv_path

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        print(f"ERROR: dataset not found at {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")

    # Normalize columns: accept either header present or no header (3 columns)
    expected = ["ClassId", "Title", "Description"]
    if set(expected).issubset(df.columns):
        df = df.rename(columns={c: c for c in expected})
    elif df.shape[1] == 3:
        df.columns = expected
    else:
        print("ERROR: Unexpected CSV format. Expected 3 columns (Class, Title, Description).")
        print("Found columns:", df.columns.tolist())
        sys.exit(1)

    # Clean and combine text
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    df["text"] = (df["Title"] + " " + df["Description"]).str.strip()
    # Drop rows with empty text or missing class
    df = df[df["text"].str.len() > 0].copy()
    df = df.dropna(subset=["ClassId"])
    return df

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    csv_path = ensure_data_extracted(data_dir=data_dir)

    df = load_dataset(csv_path)
    if df.empty:
        print("ERROR: No data found after cleaning.")
        sys.exit(1)

    X = df["text"]
    y = df["ClassId"].astype(str)  # ensure labels are strings

    counts = y.value_counts()
    print("Class distribution:\n", counts.to_dict())
    min_count = int(counts.min())

    # If any class has fewer than 2 samples, we cannot stratify.
    if min_count < 2:
        print(f"⚠️ Some classes have fewer than 2 samples (min={min_count}). Disabling stratify.")
        stratify_param = None
    else:
        stratify_param = y

    # Perform split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    # Vectorize and train
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "nb_model.pkl"))
    joblib.dump(tfidf, os.path.join("models", "tfidf.pkl"))
    print("✅ Model and TF-IDF saved successfully in models/ folder.")

if __name__ == "__main__":
    main()
# ...existing code...