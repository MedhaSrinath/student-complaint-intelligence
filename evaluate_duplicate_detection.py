import csv
from pathlib import Path
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LABELED_PAIRS = [
    (
        "Hostel Wi-Fi keeps disconnecting and students cannot access internet.",
        "The wireless connection in the dorm drops repeatedly and residents lose internet access.",
        True,
    ),
    (
        "A water pipe is leaking in the hostel bathroom and flooding the floor.",
        "Water continues to spill from a broken pipe in the residence washroom.",
        True,
    ),
    (
        "The campus shuttle is consistently arriving 30 minutes late.",
        "The university bus service has been delayed by half an hour.",
        True,
    ),
    (
        "The classroom projector will not turn on during lectures.",
        "The lecture room's projection equipment is not powering up.",
        True,
    ),
    (
        "The online fee portal charged me twice for semester tuition.",
        "My tuition payment was processed two times by the university website.",
        True,
    ),
    (
        "The air conditioner in the hostel room is broken and blows hot air.",
        "Our dorm room AC has stopped cooling and only warm air comes out.",
        True,
    ),
    (
        "The library printer jams and will not print student documents.",
        "The printer in the library keeps getting paper stuck and cannot print.",
        True,
    ),
    (
        "Cafeteria meals are served cold to students.",
        "Food in the dining hall is arriving cold instead of hot.",
        True,
    ),
    (
        "The course registration website crashes whenever I add a class.",
        "I cannot enroll in courses because the registration portal keeps crashing.",
        True,
    ),
    (
        "Streetlights near the main gate are not working after dark.",
        "The lamps by the campus entrance stay off at night.",
        True,
    ),
    (
        "Hostel Wi-Fi keeps disconnecting in the dorm.",
        "The hostel bathroom pipe is leaking and flooding the floor.",
        False,
    ),
    (
        "The campus shuttle arrives late every morning.",
        "The campus cafeteria serves cold food at lunch.",
        False,
    ),
    (
        "The classroom projector will not turn on.",
        "The classroom air conditioner is broken and blowing warm air.",
        False,
    ),
    (
        "The library printer keeps jamming.",
        "The library is too noisy for students to study.",
        False,
    ),
    (
        "My tuition payment was charged twice online.",
        "I need a scholarship application form for next semester.",
        False,
    ),
    (
        "The hostel room AC has stopped cooling.",
        "The hostel washing machine is broken and will not spin.",
        False,
    ),
    (
        "The course registration portal crashes when adding classes.",
        "The exam timetable has the wrong date for my course.",
        False,
    ),
    (
        "The security light near the campus gate is broken.",
        "The security guard at the main gate was rude to visitors.",
        False,
    ),
    (
        "The water pipe in the hostel bathroom is leaking.",
        "The water dispenser in the cafeteria is empty.",
        False,
    ),
    (
        "Students cannot access the internet because hostel Wi-Fi is down.",
        "I was charged an extra fee for staying in the hostel.",
        False,
    ),
]

THRESHOLDS = (15, 20, 25, 30, 35, 40, 45, 50)
DATASET_PATH = Path(__file__).parent / "dataset" / "complaints.csv"


def evaluate_thresholds(
    labels: Sequence[bool],
    similarities: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, float | int]]:
    if len(labels) != len(similarities):
        raise ValueError("labels and similarities must have the same length")

    results = []
    for threshold in thresholds:
        true_positive = false_positive = false_negative = true_negative = 0
        for label, similarity in zip(labels, similarities):
            predicted = similarity * 100 >= threshold
            if predicted and label:
                true_positive += 1
            elif predicted:
                false_positive += 1
            elif label:
                false_negative += 1
            else:
                true_negative += 1

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        })

    return results


def run_evaluation() -> list[dict[str, float | int]]:
    with DATASET_PATH.open(encoding="utf-8-sig", newline="") as dataset_file:
        dataset_texts = [row["text"] for row in csv.DictReader(dataset_file)]

    pair_texts = [text for first, second, _ in LABELED_PAIRS for text in (first, second)]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=15000,
    )
    vectorizer.fit(dataset_texts)
    pair_vectors = vectorizer.transform(pair_texts)
    similarities = cosine_similarity(
        pair_vectors[::2],
        pair_vectors[1::2],
    ).diagonal()
    labels = [is_duplicate for _, _, is_duplicate in LABELED_PAIRS]
    return evaluate_thresholds(labels, similarities, THRESHOLDS)


if __name__ == "__main__":
    results = run_evaluation()
    print(f"Labeled pairs: {len(LABELED_PAIRS)}")
    print(f"Positive pairs: {sum(label for _, _, label in LABELED_PAIRS)}")
    print(f"Negative pairs: {sum(not label for _, _, label in LABELED_PAIRS)}")
    print(
        "\nThreshold  Precision  Recall  F1     TP  FP  FN  TN\n"
        + "-" * 55
    )
    for result in results:
        print(
            f"{result['threshold']:>9}%  "
            f"{result['precision']:>9.2f}  "
            f"{result['recall']:>6.2f}  "
            f"{result['f1']:>5.2f}  "
            f"{result['true_positive']:>2}  "
            f"{result['false_positive']:>2}  "
            f"{result['false_negative']:>2}  "
            f"{result['true_negative']:>2}"
        )
    print("\nThese results are for a small hand-labeled evaluation set.")
