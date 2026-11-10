
from transformers import pipeline

sentiment_analyzer = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def analyze_sentiment(text):
    result = sentiment_analyzer(text[:512])[0]

    labels = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral"
    }

    return {
        "sentiment": labels.get(result["label"].lower(), result["label"]),
        "confidence": round(result["score"] * 100, 2)
    }


if __name__ == "__main__":
    complaints = [
        "The hostel Wi-Fi is not working again. This is extremely frustrating.",
        "Please install another water dispenser near the library.",
        "Thank you for fixing the classroom projector. It works perfectly now.",
        "The classroom projector is not working. Please arrange a technician."
    ]

    for complaint in complaints:
        result = analyze_sentiment(complaint)

        print("\nComplaint:", complaint)
        print("Sentiment:", result["sentiment"])
        print("Confidence:", result["confidence"], "%")
