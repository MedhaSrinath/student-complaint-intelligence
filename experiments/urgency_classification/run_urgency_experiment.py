import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Paths
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_candidate_urgency_severity.csv"
EXP_DIR = Path(__file__).resolve().parent
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "severity"]).reset_index(drop=True)
X = df["text"].astype(str).values
y = df["severity"].astype(str).values
groups = df["group_id"].values if "group_id" in df.columns else np.arange(len(df))

# Build pipelines
def build_pipeline(model_name: str):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    elif model_name == "svm":
        clf = LinearSVC(class_weight="balanced")
    else:
        raise ValueError("unknown model")
    return make_pipeline(vec, clf)

models = {"logreg": build_pipeline("logreg"), "svm": build_pipeline("svm")}

# Helper to evaluate a CV splitter
def evaluate(cv_splitter, split_name):
    pooled = {name: {"y_true": [], "y_pred": []} for name in models}
    fold_metrics = {name: [] for name in models}
    confusion_matrices = {name: [] for name in models}
    misclassifications = {name: [] for name in models}

    for name, pipe in models.items():
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # Fit inside CV
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            # Record pooled predictions
            pooled[name]["y_true"].extend(y_test.tolist())
            pooled[name]["y_pred"].extend(preds.tolist())
            # Fold metrics
            acc = accuracy_score(y_test, preds)
            macro_f1 = f1_score(y_test, preds, average="macro")
            precision = precision_score(y_test, preds, average="macro", zero_division=0)
            recall = recall_score(y_test, preds, average="macro", zero_division=0)
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            per_class = {cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"], "f1": report[cls]["f1-score"]} for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
            fold_metrics[name].append({
                "fold": fold,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "precision": precision,
                "recall": recall,
                "per_class": per_class,
            })
            # Confusion matrix
            cm = confusion_matrix(y_test, preds, labels=pipe.named_steps[pipe.steps[-1][0]].classes_)
            confusion_matrices[name].append(cm.tolist())
            # Misclassifications
            for i, (true, pred) in enumerate(zip(y_test, preds)):
                if true != pred:
                    misclassifications[name].append({
                        "text": X_test[i],
                        "true_label": true,
                        "pred_label": pred,
                        "fold": fold,
                    })
    # Aggregate pooled metrics
    overall_metrics = {}
    for name in models:
        y_true = np.array(pooled[name]["y_true"])
        y_pred = np.array(pooled[name]["y_pred"])
        overall = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        per_class = {cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"], "f1": report[cls]["f1-score"]} for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
        overall["per_class"] = per_class
        overall_metrics[name] = overall
    # Save results
    results = {
        "split": split_name,
        "overall_metrics": overall_metrics,
        "mean_fold_metrics": fold_metrics,
        "confusion_matrices": confusion_matrices,
        "misclassifications": misclassifications,
    }
    results_path = EXP_DIR / f"results_{split_name}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"{split_name} evaluation completed. Results written to", results_path)

# Standard stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
evaluate(skf, "standard")
# Duplicate-aware GroupKFold (using group_id)
gkf = GroupKFold(n_splits=5)
evaluate(gkf, "duplicate_aware")

# Train final models on full data and save pipelines
for name, pipe in models.items():
    pipe.fit(X, y)
    model_path = EXP_DIR / f"{name}_pipeline.pkl"
    joblib.dump(pipe, model_path)
    print(f"Trained full {name} model saved to", model_path)
