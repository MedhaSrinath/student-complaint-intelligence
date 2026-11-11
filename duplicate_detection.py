import re
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_TOPIC_STOP_WORDS = ENGLISH_STOP_WORDS.union({
    "broken", "campus", "complaint", "fail", "fails", "failure",
    "issue", "need", "please", "problem", "stopped", "student",
    "students", "university", "work", "working",
})


def _build_topic_label(
    complaint_texts: Sequence[str],
    vectorizer: TfidfVectorizer,
) -> str:
    vectors = vectorizer.transform(complaint_texts)
    weights = np.asarray(vectors.sum(axis=0)).ravel()
    features = vectorizer.get_feature_names_out()
    representative = re.sub(
        r"\bwi[\s-]*fi\b",
        "wifi",
        complaint_texts[0].lower(),
    )
    candidates = {}

    for index in weights.argsort()[::-1]:
        if weights[index] <= 0:
            break

        feature = features[index]
        tokens = re.findall(r"[a-z]+", feature.lower())
        tokens = [
            "wifi" if token in {"wi", "fi"} else token
            for token in tokens
        ]
        tokens = [
            token for token in tokens
            if token not in _TOPIC_STOP_WORDS
            and (len(token) >= 3 or token == "wifi")
        ]
        tokens = list(dict.fromkeys(tokens))

        if not tokens:
            continue

        label = " ".join(tokens)
        position = representative.find(label)
        if position < 0:
            position = len(representative)

        candidate = (float(weights[index]), len(tokens), -position)
        if label not in candidates or candidate > candidates[label]:
            candidates[label] = candidate

    if not candidates:
        return "Recurring issue"

    label = max(
        candidates,
        key=lambda value: (
            candidates[value][0],
            candidates[value][1],
            candidates[value][2],
        ),
    )
    return " ".join(token.title() for token in label.split()).replace(
        "Wifi",
        "Wi-Fi",
    )


def find_similar_submitted_complaints(
    text: str,
    complaints: Sequence[Mapping[str, Any]],
    vectorizer: TfidfVectorizer,
    top_n: int = 3,
    min_similarity: float = 20.0,
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


def find_recurring_complaint_groups(
    complaints: Sequence[Mapping[str, Any]],
    vectorizer: TfidfVectorizer,
    min_similarity: float = 20.0,
) -> list[dict[str, Any]]:
    if not 0 <= min_similarity <= 100:
        raise ValueError("min_similarity must be between 0 and 100")

    remaining = [
        complaint
        for complaint in complaints
        if str(complaint.get("complaint_text", "")).strip()
    ]
    groups = []

    while remaining:
        representative = remaining.pop(0)
        representative_text = str(representative["complaint_text"])
        members = [{
            "id": representative.get("id"),
            "complaint": representative_text,
            "status": representative.get("status", "Pending"),
            "support_count": int(representative.get("support_count", 0) or 0),
            "similarity": 100.0,
        }]
        unmatched = []

        if remaining:
            query_vector = vectorizer.transform([representative_text])
            candidate_vectors = vectorizer.transform([
                str(complaint["complaint_text"]) for complaint in remaining
            ])
            scores = cosine_similarity(query_vector, candidate_vectors)[0]

            for complaint, score in zip(remaining, scores):
                similarity = round(float(score) * 100, 2)
                if similarity >= min_similarity:
                    members.append({
                        "id": complaint.get("id"),
                        "complaint": complaint["complaint_text"],
                        "status": complaint.get("status", "Pending"),
                        "support_count": int(
                            complaint.get("support_count", 0) or 0
                        ),
                        "similarity": similarity,
                    })
                else:
                    unmatched.append(complaint)

        report_count = sum(
            1 + member["support_count"] for member in members
        )
        member_texts = [member["complaint"] for member in members]
        groups.append({
            "representative": representative_text,
            "topic_label": _build_topic_label(member_texts, vectorizer),
            "count": report_count,
            "complaints": members,
        })

        remaining = unmatched

    return sorted(groups, key=lambda group: group["count"], reverse=True)
