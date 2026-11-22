import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)

# Paths
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_candidate_genres_omarsobhy.csv"
EXP_DIR = Path(__file__).resolve().parent
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
# Ensure no missing values in required columns
df = df.dropna(subset=["text", "genre"]).reset_index(drop=True)
X = df["text"].astype(str).values
y = df["genre"].astype(str).values

# Build pipelines for models
def build_pipeline(name):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    if name == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    elif name == "svm":
        clf = LinearSVC(class_weight="balanced")
    else:
        raise ValueError("unknown model")
    return make_pipeline(vec, clf)

models = {"logreg": build_pipeline("logreg"), "svm": build_pipeline("svm")}

# Cross‑validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Containers
overall_metrics = {}
confusion_matrices = {name: [] for name in models}
fold_metrics = {name: [] for name in models}
misclassifications = {name: [] for name in models}
out_of_fold = {name: [] for name in models}

for name, pipe in models.items():
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        # Record predictions for reproducibility
        for i, (txt, true, pred) in enumerate(zip(X_test, y_test, preds)):
            out_of_fold[name].append({"fold": fold, "text": txt, "true_label": true, "pred_label": pred})
        # Metrics
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro", zero_division=0)
        recall = recall_score(y_test, preds, average="macro", zero_division=0)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        per_class = {cls: {"precision": report[cls]["precision"],
                           "recall": report[cls]["recall"],
                           "f1": report[cls]["f1-score"]}
                     for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")}
        fold_metrics[name].append({"fold": fold, "accuracy": acc, "macro_f1": macro_f1,
                                   "precision": precision, "recall": recall,
                                   "per_class": per_class})
        # Confusion matrix (order matches pipe.named_steps[pipe.steps[-1][0]].classes_)
        cm = confusion_matrix(y_test, preds, labels=pipe.named_steps[pipe.steps[-1][0]].classes_)
        confusion_matrices[name].append(cm.tolist())
        # Misclassifications
        for txt, true, pred in zip(X_test, y_test, preds):
            if true != pred:
                misclassifications[name].append({"fold": fold, "text": txt,
                                                "true_label": true, "pred_label": pred})
    # Aggregate overall metrics across folds
    df_fold = pd.DataFrame(fold_metrics[name])
    numeric_cols = df_fold.select_dtypes(include='number').columns
    overall_metrics[name] = df_fold[numeric_cols].mean().to_dict()
    # Fit final model on all data and save pipeline
    pipe.fit(X, y)
    model_path = EXP_DIR / f"{name}_pipeline.pkl"
    import joblib
    joblib.dump(pipe, model_path)

# Save results
results = {
    "overall_metrics": overall_metrics,
    "confusion_matrices": confusion_matrices,
    "misclassifications": misclassifications,
    "fold_metrics": fold_metrics
}
with open(EXP_DIR / "results_genre.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
# Save out‑of‑fold predictions
with open(EXP_DIR / "predictions_genre.json", "w", encoding="utf-8") as f:
    json.dump(out_of_fold, f, indent=2)
print("Experiment completed. Results saved to", EXP_DIR)
