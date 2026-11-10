
import joblib
import re

urgency_model = joblib.load("models/urgency_model.pkl")

critical_keywords = [
    "fire",
    "smoke",
    "gas leak",
    "electrical sparks",
    "exposed wire",
    "live wire",
    "electric shock",
    "flooding electrical room",
    "ceiling collapse",
    "structural collapse",
    "medical emergency"
]

time_sensitive_keywords = [
    "urgent",
    "immediately",
    "right now",
    "today",
    "deadline",
    "exam closes",
    "in progress",
    "time sensitive",
    "emergency"
]


def contains_keyword(text, keywords):
    text = text.lower()

    return any(
        re.search(r"\b" + re.escape(keyword) + r"\b", text)
        for keyword in keywords
    )


def calculate_priority(text):

    probabilities = urgency_model.predict_proba([text])[0]
    classes = urgency_model.classes_

    urgency_scores = {
        "Low": 10,
        "Medium": 35,
        "High": 65
    }

    urgency_score = sum(
        probabilities[i] * urgency_scores.get(classes[i], 0)
        for i in range(len(classes))
    )

    is_critical = contains_keyword(text, critical_keywords)
    is_time_sensitive = contains_keyword(text, time_sensitive_keywords)

    priority_score = urgency_score

    if is_time_sensitive:
        priority_score += 10

    priority_score = min(round(priority_score, 2), 100)

    if is_critical:
        priority = "Critical"
    elif priority_score >= 50:
        priority = "High"
    elif priority_score >= 25:
        priority = "Medium"
    else:
        priority = "Low"

    predicted_urgency = classes[probabilities.argmax()]
    urgency_confidence = round(max(probabilities) * 100, 2)

    needs_review = (
        priority in ["Critical", "High"]
        or urgency_confidence < 55
    )

    return {
        "predicted_urgency": predicted_urgency,
        "urgency_confidence": urgency_confidence,
        "priority_score": priority_score,
        "priority": priority,
        "critical_indicator": is_critical,
        "time_sensitive": is_time_sensitive,
        "needs_human_review": needs_review
    }


if __name__ == "__main__":

    test_complaints = [
        "The hostel Wi-Fi is slow and keeps disconnecting.",
        "The library projector is not working. Please fix it today.",
        "There are electrical sparks coming from the classroom switchboard.",
        "My examination portal is not working and my exam closes in 45 minutes.",
        "Please install another water dispenser near the library."
    ]

    print("\n========== CAMPUSPULSE PRIORITY ANALYSIS ==========\n")

    for complaint in test_complaints:

        result = calculate_priority(complaint)

        print("Complaint:", complaint)
        print("Predicted Urgency:", result["predicted_urgency"])
        print("Urgency Confidence:", result["urgency_confidence"], "%")
        print("Priority Score:", result["priority_score"])
        print("Priority:", result["priority"])
        print("Critical Indicator:", result["critical_indicator"])
        print("Time Sensitive:", result["time_sensitive"])
        print("Needs Human Review:", result["needs_human_review"])
        print("-" * 65)
