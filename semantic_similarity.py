import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("dataset/complaints.csv")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

complaint_embeddings = model.encode(
    df["text"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)


def find_similar_complaints(text, top_n=5):

    query_embedding = model.encode(
        text,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        query_embedding,
        complaint_embeddings
    )[0]

    top_results = similarities.argsort(descending=True)[:top_n]

    results = []

    for index in top_results:
        index = index.item()

        results.append({
            "complaint": df.iloc[index]["text"],
            "department": df.iloc[index]["department"],
            "urgency": df.iloc[index]["urgency"],
            "similarity": round(similarities[index].item() * 100, 2)
        })

    return results


if __name__ == "__main__":

    test_complaint = (
        "The hostel Wi-Fi keeps disconnecting and "
        "students are unable to access the internet."
    )

    print("\n========== SEMANTIC SIMILARITY ==========\n")

    results = find_similar_complaints(test_complaint)

    for result in results:
        print("Complaint:", result["complaint"])
        print("Department:", result["department"])
        print("Urgency:", result["urgency"])
        print("Similarity Score:", result["similarity"], "%")
        print("-" * 65)