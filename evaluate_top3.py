
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("dataset/complaints.csv")

X = df["text"]
y = df["department"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("models/department_model.pkl")

probabilities = model.predict_proba(X_test)
classes = model.classes_

top1_predictions = model.predict(X_test)

top3_correct = 0

for i, actual in enumerate(y_test):
    top3_indices = probabilities[i].argsort()[::-1][:3]
    top3_departments = classes[top3_indices]

    if actual in top3_departments:
        top3_correct += 1

top1_accuracy = accuracy_score(y_test, top1_predictions)
top3_accuracy = top3_correct / len(y_test)

print("\n========== DEPARTMENT MODEL EVALUATION ==========")
print("Total test complaints:", len(y_test))
print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
print(f"Top-3 Accuracy: {top3_accuracy * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, top1_predictions))
