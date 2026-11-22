import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.preprocessing import normalize

# Sentence‑Transformer (assumes package is installed)
from sentence_transformers import SentenceTransformer

# ---------------------------
# Paths
# ---------------------------
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_candidate_duplicate_detection_groups.csv"
EXP_DIR = Path(__file__).resolve().parent
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load dataset
# ---------------------------
# Keep only rows with text and group_id
df = pd.read_csv(DATA_PATH).dropna(subset=["text", "group_id"]).reset_index(drop=True)
texts = df["text"].astype(str).values
group_ids = df["group_id"].values

# ---------------------------
# Helper utilities
# ---------------------------
def pairwise_indices(n):
    """Return list of (i, j) with i < j for a set of size n"""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def compute_upper_triangular_sim(matrix):
    """Given L2‑normalized row vectors, return cosine similarities for i<j"""
    sim = np.dot(matrix, matrix.T)
    idx = np.triu_indices_from(sim, k=1)
    return sim[idx]

def optimal_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1)
    # thresholds length = len(precision)-1; handle edge case
    if best_idx < len(thresholds):
        return thresholds[best_idx]
    else:
        return 0.5

def evaluate(y_true, scores, thresh):
    # Convert raw similarity scores to binary predictions using the given threshold
    y_pred = (scores >= thresh).astype(int)
    # Standard binary metrics
    precision = np.mean(y_pred[y_true == 1] == 1) if np.any(y_true == 1) else 0.0
    recall = np.mean(y_pred[y_true == 1] == 1) if np.any(y_true == 1) else 0.0
    f1 = f1_score(y_true, y_pred)
    # Compute PR AUC safely – precision_recall_curve may return a single point
    pr, rc, _ = precision_recall_curve(y_true, scores)
    if len(pr) > 1 and len(rc) > 1:
        pr_auc = float(auc(rc, pr))
    else:
        # No variation in scores; define PR AUC as the precision value (or 0.0 if undefined)
        pr_auc = float(pr[0]) if pr.size > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "tp": int(np.sum((y_pred == 1) & (y_true == 1))),
        "fp": int(np.sum((y_pred == 1) & (y_true == 0))),
        "fn": int(np.sum((y_pred == 0) & (y_true == 1)))
    }

# ---------------------------
# Experiment configuration
# ---------------------------
# Use GroupKFold to keep same group together across splits
gkf = GroupKFold(n_splits=5)
# Choose first split as validation for threshold selection, remaining as test
fold_indices = list(gkf.split(texts, groups=group_ids))
val_fold = fold_indices[0]
test_folds = fold_indices[1:]

# Sentence‑Transformer model (downloaded on first use)
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Containers for results
results = {
    "tfidf": {"precision": [], "recall": [], "f1": [], "pr_auc": [], "tp": 0, "fp": 0, "fn": 0},
    "sentence_transformer": {"precision": [], "recall": [], "f1": [], "pr_auc": [], "tp": 0, "fp": 0, "fn": 0}
}
# Store threshold found on validation
thresholds = {}
# Store out‑of‑fold predictions (scores and true labels) for reproducibility
predictions = {"tfidf": [], "sentence_transformer": []}

# ---------------------------
# Validation fold – compute thresholds
# ---------------------------
train_idx, val_idx = val_fold
X_train, X_val = [texts[i] for i in train_idx], [texts[i] for i in val_idx]
y_train, y_val = group_ids[train_idx], group_ids[val_idx]
# Build positive‑label vector for validation pairs
n_val = len(X_val)
pair_idx = pairwise_indices(n_val)
y_pair_val = np.array([1 if y_val[i] == y_val[j] else 0 for i, j in pair_idx])
# TF‑IDF
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
tfidf_vec.fit(X_train)
val_tfidf = tfidf_vec.transform(X_val)
val_tfidf = normalize(val_tfidf)
val_tfidf_scores = compute_upper_triangular_sim(val_tfidf.toarray())
# Sentence‑Transformer
val_st_emb = st_model.encode(X_val, show_progress_bar=False)
val_st_emb = normalize(val_st_emb)
val_st_scores = compute_upper_triangular_sim(val_st_emb)
# Determine optimal thresholds on validation data
thresholds['tfidf'] = float(optimal_threshold(y_pair_val, val_tfidf_scores))
thresholds['sentence_transformer'] = optimal_threshold(y_pair_val, val_st_scores)

# ---------------------------
# Test folds – evaluate using fixed thresholds
# ---------------------------
for fold_num, (train_idx, test_idx) in enumerate(test_folds, start=1):
    X_train_fold, X_test = [texts[i] for i in train_idx], [texts[i] for i in test_idx]
    y_train_fold, y_test = group_ids[train_idx], group_ids[test_idx]
    n_test = len(X_test)
    pair_idx = pairwise_indices(n_test)
    y_pair_test = np.array([1 if y_test[i] == y_test[j] else 0 for i, j in pair_idx])
    # TF‑IDF (fit on training portion of this fold)
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    tfidf_vec.fit(X_train_fold)
    test_tfidf = tfidf_vec.transform(X_test)
    test_tfidf = normalize(test_tfidf)
    tfidf_scores = compute_upper_triangular_sim(test_tfidf.toarray())
    tfidf_res = evaluate(y_pair_test, tfidf_scores, thresholds['tfidf'])
    results['tfidf']['precision'].append(tfidf_res['precision'])
    results['tfidf']['recall'].append(tfidf_res['recall'])
    results['tfidf']['f1'].append(tfidf_res['f1'])
    results['tfidf']['pr_auc'].append(tfidf_res['pr_auc'])
    results['tfidf']['tp'] += tfidf_res['tp']
    results['tfidf']['fp'] += tfidf_res['fp']
    results['tfidf']['fn'] += tfidf_res['fn']
    # Convert numpy float32 scores to native Python floats for JSON serialization
    predictions['tfidf'].append({"fold": fold_num, "scores": [float(s) for s in tfidf_scores.tolist()], "labels": y_pair_test.tolist()})
    # Sentence‑Transformer (no fitting needed)
    test_st_emb = st_model.encode(X_test, show_progress_bar=False)
    test_st_emb = normalize(test_st_emb)
    st_scores = compute_upper_triangular_sim(test_st_emb)
    st_res = evaluate(y_pair_test, st_scores, thresholds['sentence_transformer'])
    results['sentence_transformer']['precision'].append(st_res['precision'])
    results['sentence_transformer']['recall'].append(st_res['recall'])
    results['sentence_transformer']['f1'].append(st_res['f1'])
    results['sentence_transformer']['pr_auc'].append(st_res['pr_auc'])
    results['sentence_transformer']['tp'] += st_res['tp']
    results['sentence_transformer']['fp'] += st_res['fp']
    results['sentence_transformer']['fn'] += st_res['fn']
    predictions['sentence_transformer'].append({"fold": fold_num, "scores": [float(s) for s in st_scores.tolist()], "labels": y_pair_test.tolist()})

# ---------------------------
# Summarise final metrics
# ---------------------------
summary = {}
for name, vals in results.items():
    total_pairs = sum(len(p['labels']) for p in predictions[name])
    pos_pairs = sum(int(np.sum(p['labels'])) for p in predictions[name])
    neg_pairs = total_pairs - pos_pairs
    summary[name] = {
        "precision": float(np.mean(vals['precision'])) if vals['precision'] else None,
        "recall": float(np.mean(vals['recall'])) if vals['recall'] else None,
        "f1": float(np.mean(vals['f1'])) if vals['f1'] else None,
        "pr_auc": float(np.mean(vals['pr_auc'])) if vals['pr_auc'] else None,
        "threshold": thresholds[name],
        "positive_pairs": pos_pairs,
        "negative_pairs": neg_pairs,
        "tp": vals['tp'],
        "fp": vals['fp'],
        "fn": vals['fn']
    }

# Save results and predictions for reproducibility
out = {"summary": summary, "predictions": predictions, "thresholds": thresholds}
out_path = EXP_DIR / "results_duplicate_detection.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("Experiment completed. Results written to", out_path)
