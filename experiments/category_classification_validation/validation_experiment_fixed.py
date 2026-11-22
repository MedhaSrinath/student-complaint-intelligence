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
# Build pipelines (same as corrected experiment)
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
# Load previous misclassifications (for consistency check)
# -------------------------------------------------------------------
prev_path = Path(__file__).resolve().parents[2] / "experiments" / "category_classification_corrected" / "results_corrected.json"
with open(prev_path) as f:
    prev = json.load(f)
prev_mis_counts = {name: len(prev["misclassifications"][name]) for name in prev["misclassifications"]}

# -------------------------------------------------------------------
# Helper to run CV given a splitter (must implement split(X, y, groups=None))
# -------------------------------------------------------------------
def run_cv(splitter, name_suffix="", groups=None):
    pooled_true = {k: [] for k in models}
    pooled_pred = {k: [] for k in models}
    mis = {k: [] for k in models}
    for name, pipe in models.items():
        # enumerate folds
        split_gen = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
        for fold, (train_idx, test_idx) in enumerate(split_gen, start=1):
            pipe.fit(X[train_idx], y[train_idx])
            preds = pipe.predict(X[test_idx])
            pooled_true[name].extend(y[test_idx])
            pooled_pred[name].extend(preds)
            for i, (t, p) in enumerate(zip(y[test_idx], preds)):
                if t != p:
                    mis[name].append({"text": X[test_idx[i]], "true_label": t, "pred_label": p, "fold": fold})
    result = {}
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
            "misclassifications": mis[name],
        }
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        per_class = {cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"], "f1": report[cls]["f1-score"]}
                     for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
        overall["per_class"] = per_class
        result[name + name_suffix] = overall
    return result

# -------------------------------------------------------------------
# 1️⃣ Standard stratified 5‑fold CV
# -------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
standard_res = run_cv(skf, name_suffix="_std")

# -------------------------------------------------------------------
# 2️⃣ Duplicate‑aware Group CV (identical texts stay together)
# -------------------------------------------------------------------
# Assign a group id per unique text string
unique_texts = pd.Series(df["text"]).astype(str)
text_to_group = {txt: i for i, txt in enumerate(pd.unique(unique_texts))}
groups = np.array([text_to_group[t] for t in unique_texts])
# GroupKFold requires at least n_splits groups, we have many
gkf = GroupKFold(n_splits=5)
duplicate_res = run_cv(gkf, name_suffix="_dup", groups=groups)

# -------------------------------------------------------------------
# 3️⃣ Consistency check: compare misclassification counts with previous report
# -------------------------------------------------------------------
consistency = {}
for name in models:
    consistency[name] = {
        "prev_reported": prev_mis_counts[name],
        "std_error_count": standard_res[name + "_std"]["error_count"],
        "dup_error_count": duplicate_res[name + "_dup"]["error_count"],
    }

# -------------------------------------------------------------------
# 4️⃣ Duplicate statistics
# -------------------------------------------------------------------
text_counts = df["text"].value_counts()
num_duplicate_texts = int((text_counts > 1).sum())
example_dups = text_counts[text_counts > 1].head(5).to_dict()

# -------------------------------------------------------------------
# Save full validation report
# -------------------------------------------------------------------
report = {
    "standard_cv": standard_res,
    "duplicate_aware_cv": duplicate_res,
    "consistency_check": consistency,
    "duplicate_stats": {
        "total_duplicate_texts": num_duplicate_texts,
        "examples": example_dups,
    },
}
out_path = Path(__file__).resolve().parent / "validation_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("Validation completed. Report written to", out_path)
