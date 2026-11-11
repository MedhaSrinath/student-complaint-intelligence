
import joblib
import pandas as pd
import re
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
from complaint_database import create_database, save_complaint
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duplicate_detection import (
    find_recurring_complaint_groups as group_submitted_complaints,
    find_similar_submitted_complaints as match_submitted_complaints,
)
from duration_extraction import extract_duration_mentions
from emotion_detection import detect_emotion


department_model = joblib.load("models/department_model.pkl")
urgency_model = joblib.load("models/urgency_model.pkl")

sentiment_model = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

similarity_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

df = pd.read_csv("dataset/complaints.csv")

complaint_embeddings = similarity_model.encode(
    df["text"].tolist(),
    convert_to_tensor=True
)

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=15000
)

complaint_vectors = tfidf_vectorizer.fit_transform(df["text"])


critical_keywords = [
    "fire", "smoke", "gas leak", "electrical sparks",
    "exposed wire", "live wire", "electric shock",
    "flooding electrical room", "ceiling collapse",
    "structural collapse", "medical emergency"
]

time_sensitive_keywords = [
    "urgent", "immediately", "right now", "today",
    "deadline", "exam closes", "in progress",
    "time sensitive", "emergency"
]

campus_locations = [
    "Block A", "Block B", "Block C", "Block D",
    "Block E", "Library", "Main Gate", "Admin Block",
    "Academic Block", "Hostel", "Auditorium",
    "Cafeteria", "Parking Area", "Computer Lab",
    "Science Lab", "Sports Ground"
]


def classify_department(text):

    probabilities = department_model.predict_proba([text])[0]
    classes = department_model.classes_

    top_indices = probabilities.argsort()[::-1][:3]

    return [
        {
            "department": classes[i],
            "confidence": round(probabilities[i] * 100, 2)
        }
        for i in top_indices
    ]


def analyze_urgency(text):

    probabilities = urgency_model.predict_proba([text])[0]
    classes = urgency_model.classes_

    index = probabilities.argmax()

    return {
        "urgency": classes[index],
        "confidence": round(probabilities[index] * 100, 2),
        "probabilities": {
            classes[i]: round(probabilities[i] * 100, 2)
            for i in range(len(classes))
        }
    }


def analyze_sentiment(text):

    result = sentiment_model(text[:512])[0]

    labels = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral"
    }

    return {
        "sentiment": labels.get(
            result["label"].lower(),
            result["label"]
        ),
        "confidence": round(result["score"] * 100, 2)
    }


def find_similar_complaints(text, top_n=5):

    query_embedding = similarity_model.encode(
        text,
        convert_to_tensor=True
    )

    semantic_scores = util.cos_sim(
        query_embedding,
        complaint_embeddings
    )[0]

    indices = semantic_scores.argsort(descending=True)[:top_n]

    results = []

    for index in indices:

        i = index.item()

        results.append({
            "complaint": df.iloc[i]["text"],
            "department": df.iloc[i]["department"],
            "urgency": df.iloc[i]["urgency"],
            "similarity": round(
                semantic_scores[i].item() * 100, 2
            )
        })

    return results


def find_tfidf_similarity(text, top_n=5):

    query_vector = tfidf_vectorizer.transform([text])

    scores = cosine_similarity(
        query_vector,
        complaint_vectors
    )[0]

    indices = scores.argsort()[::-1][:top_n]

    return [
        {
            "complaint": df.iloc[i]["text"],
            "department": df.iloc[i]["department"],
            "urgency": df.iloc[i]["urgency"],
            "similarity": round(scores[i] * 100, 2)
        }
        for i in indices
    ]


def find_similar_submitted_complaints(text, complaints, top_n=3, min_similarity=20.0):
    return match_submitted_complaints(
        text,
        complaints,
        tfidf_vectorizer,
        top_n=top_n,
        min_similarity=min_similarity
    )


def find_recurring_submitted_complaint_groups(complaints):
    return group_submitted_complaints(
        complaints,
        tfidf_vectorizer,
    )


def extract_keywords(text):

    if nlp is not None:
        try:
            doc = nlp(text)
            keywords = []
            for token in doc:
                if (
                    token.pos_ in ["NOUN", "PROPN"]
                    and not token.is_stop
                    and len(token.text) > 2
                ):
                    keywords.append(token.lemma_.lower())
            if keywords:
                return list(dict.fromkeys(keywords))
        except Exception:
            pass

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    custom_stops = set(ENGLISH_STOP_WORDS).union({
        "please", "kindly", "request", "issue", "problem", "complaint"
    })
    keywords = [w for w in words if w not in custom_stops]
    return list(dict.fromkeys(keywords))[:10]


def extract_entities(text):

    entities = []

    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                entities.append({
                    "entity": ent.text,
                    "type": ent.label_,
                    "description": spacy.explain(ent.label_) if hasattr(spacy, "explain") else ent.label_
                })
        except Exception:
            pass

    for location in campus_locations:

        pattern = r"\b" + re.escape(location) + r"\b"

        for match in re.finditer(pattern, text, re.IGNORECASE):

            entities.append({
                "entity": match.group(),
                "type": "CAMPUS_LOCATION",
                "description": "Recognised campus location"
            })

    entities.extend(extract_duration_mentions(text))

    unique_entities = []
    seen = set()

    for entity in entities:

        key = (entity["entity"].lower(), entity["type"])

        if key not in seen:
            unique_entities.append(entity)
            seen.add(key)

    return unique_entities


def calculate_priority(text, urgency_result):

    probabilities = urgency_result["probabilities"]

    base_scores = {
        "Low": 10,
        "Medium": 35,
        "High": 65
    }

    score = sum(
        probabilities[label] * base_scores[label] / 100
        for label in base_scores
    )

    critical = any(
        keyword in text.lower()
        for keyword in critical_keywords
    )

    time_sensitive = any(
        keyword in text.lower()
        for keyword in time_sensitive_keywords
    )

    if time_sensitive:
        score += 10

    score = min(round(score, 2), 100)

    if critical:
        priority = "Critical"
    elif score >= 50:
        priority = "High"
    elif score >= 25:
        priority = "Medium"
    else:
        priority = "Low"

    return {
        "score": score,
        "priority": priority,
        "critical_indicator": critical,
        "time_sensitive": time_sensitive,
        "needs_human_review": (
            priority in ["Critical", "High"]
            or urgency_result["confidence"] < 55
        )
    }


def process_complaint(text):

    departments = classify_department(text)
    urgency = analyze_urgency(text)
    sentiment = analyze_sentiment(text)
    emotion = detect_emotion(text)
    priority = calculate_priority(text, urgency)
    keywords = extract_keywords(text)
    entities = extract_entities(text)
    semantic_matches = find_similar_complaints(text)
    tfidf_matches = find_tfidf_similarity(text)

    return {
        "department_predictions": departments,
        "urgency": urgency,
        "sentiment": sentiment,
        "emotion": emotion,
        "priority": priority,
        "keywords": keywords,
        "entities": entities,
        "semantic_similar_complaints": semantic_matches,
        "tfidf_similar_complaints": tfidf_matches
    }


if __name__ == "__main__":

    complaint = (
        "The university shuttle is consistently arriving 30 minutes "
        "late, causing students to miss their first lecture."
    )

    result = process_complaint(complaint)

    create_database()
    complaint_id = save_complaint(complaint, result)

    print(f"\nComplaint saved successfully! ID: {complaint_id}")

    print("\n========== CAMPUSPULSE NLP PIPELINE ==========\n")

    for section, values in result.items():

        print(f"\n{section.upper()}")

        if isinstance(values, list):

            for item in values:
                print(item)

        else:
            print(values)