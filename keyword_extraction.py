
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

df = pd.read_csv("dataset/complaints.csv")

custom_stopwords = set(ENGLISH_STOP_WORDS).union({
    "student", "students", "university", "campus", "college",
    "please", "kindly", "request", "requested",
    "complaint", "issue", "problem", "help",
    "department", "today", "yesterday",
    "need", "needed", "want", "would",
    "could", "should", "get", "got",
    "make", "made", "also", "really",
    "very", "much", "many", "every",
    "working", "happening", "trying"
    "like", "just", "bit", "says", "don", "ve",
    "able", "actually", "really", "small"
})

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words=list(custom_stopwords),
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.80,
    max_features=5000,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)

vectors = vectorizer.fit_transform(df["text"])

terms = vectorizer.get_feature_names_out()
frequencies = vectors.sum(axis=0).A1

keyword_df = pd.DataFrame({
    "Keyword / Phrase": terms,
    "Frequency": frequencies
}).sort_values("Frequency", ascending=False)


def get_top_keywords(data, top_n=10):

    matrix = vectorizer.transform(data["text"])
    counts = matrix.sum(axis=0).A1

    results = pd.DataFrame({
        "Keyword / Phrase": terms,
        "Frequency": counts
    })

    results = results[results["Frequency"] > 0]

    return results.sort_values(
        by="Frequency",
        ascending=False
    ).head(top_n)

if __name__ == "__main__":

    print("\n========== TOP RECURRING KEYWORDS ==========\n")
    print(keyword_df.head(20).to_string(index=False))

    print("\n========== DEPARTMENT-WISE KEYWORDS ==========\n")

    for department in sorted(df["department"].unique()):

        department_data = df[df["department"] == department]

        print(f"\nDepartment: {department}")

        print(
            get_top_keywords(department_data, 10)
            .to_string(index=False)
        )
