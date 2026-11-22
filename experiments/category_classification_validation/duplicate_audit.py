import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report

# Load data
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_candidate_category_alaminxpro.csv"
df = pd.read_csv(DATA_PATH).dropna(subset=["text", "category"]).reset_index(drop=True)
X = df["text"].astype(str).values
y = df["category"].astype(str).values
n_samples = len(df)
print(f"Total samples: {n_samples}")

# Build pipelines (same as earlier)
def build_pipe(name):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    if name == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    else:
        clf = LinearSVC(class_weight="balanced")
    return make_pipeline(vec, clf)

models = {"logreg": build_pipe("logreg"), "svm": build_pipe("svm")}

# Assign groups per unique text
unique_texts = pd.Series(df["text"]).astype(str)
text_to_group = {txt: i for i, txt in enumerate(pd.unique(unique_texts))}
groups = np.array([text_to_group[t] for t in unique_texts])

# GroupKFold (5 splits)
gkf = GroupKFold(n_splits=5)

audit = {}
for name, pipe in models.items():
    y_true_all = []
    y_pred_all = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(preds)
    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=pipe.named_steps[pipe.steps[-1][0]].classes_)
    correct = int((y_true_arr == y_pred_arr).sum())
    errors = len(y_true_arr) - correct
    audit[name] = {
        "total_samples": int(len(y_true_arr)),
        "correct": correct,
        "errors": errors,
        "confusion_matrix": cm.tolist(),
        "confusion_sum": int(cm.sum()),
        "off_diagonal_sum": int(cm.sum() - np.trace(cm)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average='macro')),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)),
        "per_class": classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
    }

out_path = Path(__file__).resolve().parent / "duplicate_audit_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
print("Audit completed. Report →", out_path)
