import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix)

# Paths
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_candidate_sentiment_crowdflower.csv"
EXP_DIR = Path(__file__).resolve().parent
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
# keep required columns
X = df["text"].astype(str).values
y = df["sentiment"].astype(str).values
original_ids = df["original_id"].values

# Build pipelines (same as other experiments)
def build_pipeline(model_name: str):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    elif model_name == "svm":
        clf = LinearSVC(class_weight="balanced")
    else:
        raise ValueError("unknown model")
    return make_pipeline(vec, clf)

models = {"logreg": build_pipeline("logreg"), "svm": build_pipeline("svm")}

# Evaluation (5‑fold stratified CV, leakage‑free)

def evaluate(cv_splitter, split_name):
    pooled = {name: {"y_true": [], "y_pred": []} for name in models}
    fold_metrics = {name: [] for name in models}
    confusion_matrices = {name: [] for name in models}
    misclassifications = {name: [] for name in models}
    all_predictions = []

    for name, pipe in models.items():
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            # pooled for overall metrics
            pooled[name]["y_true"].extend(y_test.tolist())
            pooled[name]["y_pred"].extend(preds.tolist())

            # per‑fold metrics
            acc = accuracy_score(y_test, preds)
            macro_f1 = f1_score(y_test, preds, average="macro")
            precision = precision_score(y_test, preds, average="macro", zero_division=0)
            recall = recall_score(y_test, preds, average="macro", zero_division=0)
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            per_class = {cls: {"precision": report[cls]["precision"],
                               "recall": report[cls]["recall"],
                               "f1": report[cls]["f1-score"]}
                         for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
            fold_metrics[name].append({
                "fold": fold,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "precision": precision,
                "recall": recall,
                "per_class": per_class,
            })

            # confusion matrix for this fold
            cm = confusion_matrix(y_test, preds, labels=pipe.named_steps[pipe.steps[-1][0]].classes_)
            confusion_matrices[name].append(cm.tolist())

            # misclassifications (record all folds)
            for i, (true, pred) in enumerate(zip(y_test, preds)):
                if true != pred:
                    misclassifications[name].append({
                        "original_id": int(original_ids[test_idx[i]]),
                        "text": X_test[i],
                        "true_label": true,
                        "pred_label": pred,
                        "fold": fold,
                    })
                # store every prediction for verification
                all_predictions.append({
                    "original_id": int(original_ids[test_idx[i]]),
                    "fold": fold,
                    "true_label": true,
                    "pred_label": pred,
                    "text": X_test[i],
                })

    # Overall metrics
    overall_metrics = {}
    for name in models:
        y_true = pooled[name]["y_true"]
        y_pred = pooled[name]["y_pred"]
        overall = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        per_class = {cls: {"precision": report[cls]["precision"],
                           "recall": report[cls]["recall"],
                           "f1": report[cls]["f1-score"]}
                     for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
        overall["per_class"] = per_class
        overall_metrics[name] = overall

    results = {
        "split": split_name,
        "overall_metrics": overall_metrics,
        "mean_fold_metrics": fold_metrics,
        "confusion_matrices": confusion_matrices,
        "misclassifications": misclassifications,
        "predictions": all_predictions,
    }
    results_path = EXP_DIR / f"results_{split_name}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"{split_name} evaluation completed. Results written to", results_path)

if __name__ == "__main__":
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    evaluate(skf, "sentiment")
