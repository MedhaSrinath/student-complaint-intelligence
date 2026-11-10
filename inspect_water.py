
import pandas as pd

df = pd.read_csv("dataset/complaints.csv")

water = df[df["text"].str.contains(
    "water|leak|pipe|plumb|sewage|drain|geyser",
    case=False,
    na=False
)]

print("TOTAL MATCHING COMPLAINTS:", len(water))
print("\nDEPARTMENT AND URGENCY DISTRIBUTION:")
print(pd.crosstab(water["department"], water["urgency"]))

print("\nALL MATCHING COMPLAINTS:")
for _, row in water.iterrows():
    print("\nComplaint:", row["text"])
    print("Department:", row["department"])
    print("Urgency:", row["urgency"])
    print("-" * 70)
