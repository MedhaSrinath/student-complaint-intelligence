import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib

# Paths
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_candidate_category_alaminxpro.csv"
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
X = df["text"].astype(str).values
y = df["category"].astype(str).values

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Models
models = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
    "svm": LinearSVC(class_weight="balanced")
}

# Cross‑validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

overall_results = {}
misclassified_records = {"logreg": [], "svm": []}
conf_matrices = {"logreg": [], "svm": []}

for name, clf in models.items():
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_vec, y), start=1):
        X_train, X_test = X_vec[train_idx], X_vec[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        # Metrics
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro", zero_division=0)
        recall = recall_score(y_test, preds, average="macro", zero_division=0)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        # Store per‑class metrics
        per_class = {cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"], "f1": report[cls]["f1-score"]}
                     for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
        fold_metrics.append({
            "fold": fold,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "per_class": per_class
        })
        # Confusion matrix
        cm = confusion_matrix(y_test, preds, labels=clf.classes_)
        conf_matrices[name].append(cm.tolist())
        # Misclassified samples
        for i, (true, pred) in enumerate(zip(y_test, preds)):
            if true != pred:
                misclassified_records[name].append({
                    "text": X[test_idx[i]],
                    "true_label": true,
                    "pred_label": pred,
                    "fold": fold
                })
    # Aggregate metrics across folds
    df_metrics = pd.DataFrame(fold_metrics)
    # Compute overall metrics for numeric columns only (exclude per_class dicts)
    numeric_cols = df_metrics.select_dtypes(include='number').columns
    overall = df_metrics[numeric_cols].mean().to_dict()
    overall_results[name] = overall
    # Train final model on all data and save
    clf.fit(X_vec, y)
    model_path = EXPERIMENT_DIR / f"{name}_model.pkl"
    joblib.dump({"model": clf, "vectorizer": vectorizer}, model_path)

# Save results
results_path = EXPERIMENT_DIR / "results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({"overall_metrics": overall_results,
               "confusion_matrices": conf_matrices,
               "misclassifications": misclassified_records}, f, indent=2)

print("Experiment completed. Results written to", results_path)
