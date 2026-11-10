
import pandas as pd
import numpy as np
import joblib
import os
os.makedirs("models", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("dataset/complaints.csv")

df = df.dropna(subset=["text", "department", "urgency"])

X = df["text"].astype(str)

targets = {
    "department": df["department"],
    "urgency": df["urgency"]
}

for target_name, y in targets.items():

    print("\n" + "=" * 55)
    print(f"TRAINING {target_name.upper()} CLASSIFIER")
    print("=" * 55)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=15000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel:", target_name)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    labels = model.classes_

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=labels))

    filename = f"models/{target_name}_model.pkl"

    joblib.dump(model, filename)

    print("\nModel saved:", filename)

print("\nALL MODELS TRAINED SUCCESSFULLY!")
