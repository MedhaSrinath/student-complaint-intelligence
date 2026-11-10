
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/complaints.csv")

print("\n========== DEPARTMENT DISTRIBUTION ==========")
print(df["department"].value_counts())

print("\n========== URGENCY DISTRIBUTION ==========")
print(df["urgency"].value_counts())

print("\n========== UNIQUE DEPARTMENTS ==========")
print(df["department"].unique())

print("\n========== UNIQUE URGENCY LEVELS ==========")
print(df["urgency"].unique())

print("\n========== COMPLAINT LENGTH ==========")
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

print(df["word_count"].describe())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

df["department"].value_counts().plot(
    kind="bar", ax=axes[0], title="Complaints by Department"
)

df["urgency"].value_counts().plot(
    kind="bar", ax=axes[1], title="Complaints by Urgency"
)

plt.tight_layout()
plt.show()
