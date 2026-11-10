
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("dataset/complaints.csv")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=15000
)

complaint_vectors = vectorizer.fit_transform(df["text"])


def find_similar_complaints(text, top_n=5):
    query_vector = vectorizer.transform([text])

    similarities = cosine_similarity(
        query_vector,
        complaint_vectors
    )[0]

    top_indices = similarities.argsort()[::-1][:top_n]

    results = []

    for index in top_indices:
        results.append({
            "complaint": df.iloc[index]["text"],
            "department": df.iloc[index]["department"],
            "urgency": df.iloc[index]["urgency"],
            "similarity": round(similarities[index] * 100, 2)
        })

    return results


if __name__ == "__main__":
    test_complaint = (
        "The hostel Wi-Fi keeps disconnecting and "
        "students are unable to access the internet."
    )

    print("\n========== SIMILAR COMPLAINTS ==========\n")

    results = find_similar_complaints(test_complaint)

    for result in results:
        print("Complaint:", result["complaint"])
        print("Department:", result["department"])
        print("Urgency:", result["urgency"])
        print("Similarity:", result["similarity"], "%")
        print("-" * 65)
