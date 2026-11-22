import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
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

# ---------------------------
# Paths
# ---------------------------
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_candidate_category_alaminxpro.csv"
EXP_DIR = Path(__file__).resolve().parent
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(DATA_PATH)
# Ensure no NaNs in text or label
df = df.dropna(subset=["text", "category"]).reset_index(drop=True)
X = df["text"].astype(str).values
y = df["category"].astype(str).values
n_samples = len(df)
print(f"Total samples after cleaning: {n_samples}")

# ---------------------------
# Helper to build model pipeline
# ---------------------------
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

# ---------------------------
# Cross‑validation
# ---------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Containers for pooled results
pooled = {name: {"y_true": [], "y_pred": []} for name in models}
# Store per‑fold metrics
fold_metrics = {name: [] for name in models}
# Store per‑fold confusion matrices
conf_matrices = {name: [] for name in models}
# Store mis‑classifications per model
misclassifications = {name: [] for name in models}

# Track duplicate texts across folds
duplicate_texts = set()
text_to_indices = {}
for idx, txt in enumerate(X):
    txt_norm = txt.strip()
    if txt_norm in text_to_indices:
        duplicate_texts.add(txt_norm)
        text_to_indices[txt_norm].append(idx)
    else:
        text_to_indices[txt_norm] = [idx]

for name, pipe in models.items():
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Fit pipeline on training data only (TF‑IDF fitted here)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        # Record pooled predictions
        pooled[name]["y_true"].extend(y_test.tolist())
        pooled[name]["y_pred"].extend(preds.tolist())
        # Metrics for this fold
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro", zero_division=0)
        recall = recall_score(y_test, preds, average="macro", zero_division=0)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        per_class = {cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"], "f1": report[cls]["f1-score"]}
                     for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
        fold_metrics[name].append({
            "fold": fold,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "per_class": per_class,
        })
        # Confusion matrix (order = pipe.named_steps[pipe.steps[-1][0]].classes_ )
        cm = confusion_matrix(y_test, preds, labels=pipe.named_steps[pipe.steps[-1][0]].classes_)
        conf_matrices[name].append(cm.tolist())
        # Mis‑classifications
        for i, (true, pred) in enumerate(zip(y_test, preds)):
            if true != pred:
                misclassifications[name].append({
                    "text": X_test[i],
                    "true_label": true,
                    "pred_label": pred,
                    "fold": fold,
                })

# ---------------------------
# Aggregate pooled metrics
# ---------------------------
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
    # per‑class metrics on pooled predictions
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = {cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"], "f1": report[cls]["f1-score"]}
                 for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
    overall["per_class"] = per_class
    overall_metrics[name] = overall

# ---------------------------
# Save results
# ---------------------------
results = {
    "overall_metrics": overall_metrics,
    "mean_fold_metrics": fold_metrics,
    "confusion_matrices": conf_matrices,
    "misclassifications": misclassifications,
    "duplicate_texts": list(duplicate_texts),
}
results_path = EXP_DIR / "results_corrected.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Train final models on full data and save pipelines
for name, pipe in models.items():
    pipe.fit(X, y)
    model_path = EXP_DIR / f"{name}_pipeline.pkl"
    joblib.dump(pipe, model_path)

print("Corrected experiment completed. Results written to", results_path)
