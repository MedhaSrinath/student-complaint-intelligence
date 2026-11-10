
import pandas as pd

df = pd.read_csv("dataset/complaints.csv")

keywords = [
    "water",
    "leak",
    "classroom",
    "bathroom",
    "repair",
    "maintenance"
]

print("\n========== KEYWORD INVESTIGATION ==========")

for keyword in keywords:

    matches = df[
        df["text"].str.contains(
            keyword,
            case=False,
            na=False
        )
    ]

    print(f"\nKeyword: {keyword}")
    print("Number of complaints:", len(matches))

    if len(matches) > 0:
        print("\nDepartment distribution:")
        print(matches["department"].value_counts())

        print("\nSample complaints:")
        for _, row in matches.head(3).iterrows():
            print("-", row["text"])
            print("  Department:", row["department"])


print("\n========== MAINTENANCE EXAMPLES ==========")

maintenance = df[df["department"] == "Maintenance"]

for complaint in maintenance["text"].head(10):
    print("-", complaint)
