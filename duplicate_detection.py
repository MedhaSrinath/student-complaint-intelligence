from typing import Any, Mapping, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_submitted_complaints(
    text: str,
    complaints: Sequence[Mapping[str, Any]],
    vectorizer: TfidfVectorizer,
    top_n: int = 3,
    min_similarity: float = 25.0,
) -> list[dict[str, Any]]:
    if not 0 <= min_similarity <= 100:
        raise ValueError("min_similarity must be between 0 and 100")

    if not text.strip() or not complaints or top_n <= 0:
        return []

    existing = [
        complaint
        for complaint in complaints
        if str(complaint.get("complaint_text", "")).strip()
    ]

    if not existing:
        return []

    query_vector = vectorizer.transform([text])
    complaint_vectors = vectorizer.transform([
        str(complaint["complaint_text"]) for complaint in existing
    ])
    scores = cosine_similarity(query_vector, complaint_vectors)[0]
    ranked_indices = scores.argsort()[::-1][:top_n]

    matches = []
    for index in ranked_indices:
        similarity = round(float(scores[index]) * 100, 2)
        if similarity < min_similarity:
            break

        complaint = existing[index]
        matches.append({
            "id": complaint.get("id"),
            "complaint": complaint["complaint_text"],
            "status": complaint.get("status", "Pending"),
            "similarity": similarity,
        })

    return matches
