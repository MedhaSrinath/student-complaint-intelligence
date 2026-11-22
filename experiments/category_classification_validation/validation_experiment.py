import json
import numpy as np
import pandas as pd
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

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_candidate_category_alaminxpro.csv"
df = pd.read_csv(DATA_PATH).dropna(subset=["text", "category"]).reset_index(drop=True)
X = df["text"].astype(str).values
y = df["category"].astype(str).values
n_samples = len(df)
print(f"Total samples: {n_samples}")

# -------------------------------------------------------------------
# Helper to build pipeline (same as corrected experiment)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Load previous misclassifications for cross‑check
# -------------------------------------------------------------------
prev_results_path = Path(__file__).resolve().parents[2] / "experiments" / "category_classification_corrected" / "results_corrected.json"
with open(prev_results_path) as f:
    prev = json.load(f)
prev_mis = {name: {(m["text"], m["true_label"], m["pred_label"]) for m in prev["misclassifications"][name]} for name in prev["misclassifications"]}

# -------------------------------------------------------------------
# Function to run CV, collect pooled predictions and misclassifications
# -------------------------------------------------------------------
def run_cv(splitter, name_suffix=""):
    pooled_true = {k: [] for k in models}
    pooled_pred = {k: [] for k in models}
    mis = {k: [] for k in models}
    for name, pipe in models.items():
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            pipe.fit(X[train_idx], y[train_idx])
            preds = pipe.predict(X[test_idx])
            pooled_true[name].extend(y[test_idx])
            pooled_pred[name].extend(preds)
            for i, (t, p) in enumerate(zip(y[test_idx], preds)):
                if t != p:
                    mis[name].append((X[test_idx[i]], t, p, fold))
    # compute metrics
    results = {}
    for name in models:
        y_true = np.array(pooled_true[name])
        y_pred = np.array(pooled_pred[name])
        cm = confusion_matrix(y_true, y_pred, labels=pipe.named_steps[pipe.steps[-1][0]].classes_)
        overall = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "error_count": int((y_true != y_pred).sum()),
            "misclassifications": [
                {"text": txt, "true_label": t, "pred_label": p, "fold": f} for (txt, t, p, f) in mis[name]
            ],
        }
        # per‑class
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        per_class = {cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"], "f1": report[cls]["f1-score"]}
                     for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
        overall["per_class"] = per_class
        results[name + name_suffix] = overall
    return results

# -------------------------------------------------------------------
# 1️⃣ Standard Stratified CV
# -------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
standard_results = run_cv(skf, name_suffix="_std")

# -------------------------------------------------------------------
# 2️⃣ Duplicate‑aware Group CV (identical texts stay together)
# -------------------------------------------------------------------
# Assign a group id per unique text
text_to_group = {txt: i for i, txt in enumerate(pd.unique(df["text"]))}
groups = np.array([text_to_group[t] for t in df["text"]])
# GroupKFold requires at least n_splits groups; we have many groups (>5)
gkf = GroupKFold(n_splits=5)
duplicate_results = run_cv(gkf, name_suffix="_dup")

# -------------------------------------------------------------------
# Compare misclassifications count to previous report
# -------------------------------------------------------------------
comparison = {}
for name in models:
    prev_count = len(prev_mis[name])
    std_count = standard_results[name + "_std"]["error_count"]
    dup_count = duplicate_results[name + "_dup"]["error_count"]
    comparison[name] = {
        "prev_reported": prev_count,
        "std_computed": std_count,
        "dup_computed": dup_count,
    }

# -------------------------------------------------------------------
# Duplicate statistics
# -------------------------------------------------------------------
duplicate_texts = df["text"].value_counts()
num_duplicates = (duplicate_texts > 1).sum()
duplicate_examples = duplicate_texts[duplicate_texts > 1].head(5).to_dict()

# -------------------------------------------------------------------
# Save all validation info
# -------------------------------------------------------------------
out = {
    "standard_cv": standard_results,
    "duplicate_aware_cv": duplicate_results,
    "misclassification_counts": comparison,
    "duplicate_stats": {
        "total_duplicate_texts": int(num_duplicates),
        "examples": duplicate_examples,
    },
}
out_path = Path(__file__).resolve().parent / "validation_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("Validation completed. Report written to", out_path)
