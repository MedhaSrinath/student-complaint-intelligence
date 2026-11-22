import json, numpy as np, pathlib, sys
path = r"c:/RVU/Sem5/student-complaint-intelligence/experiments/category_classification_corrected/results_corrected.json"
with open(path, 'r', encoding='utf-8') as f:
    res = json.load(f)
for name in ['logreg', 'svm']:
    mats = res['confusion_matrices'][name]
    total = np.sum(np.array(mats), axis=0)
    print(name)
    print(total.tolist())
