import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from duplicate_detection import find_similar_submitted_complaints


def build_vectorizer():
    complaints = [
        "The hostel Wi-Fi keeps disconnecting and students cannot access internet.",
        "The cafeteria meals are cold and food quality is poor.",
        "The library computer lab printers are not working.",
    ]
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    ).fit(complaints)


def test_finds_and_ranks_similar_submitted_complaints():
    complaints = [
        {
            "id": 1,
            "complaint_text": "The cafeteria meals are cold and food quality is poor.",
            "status": "Pending",
        },
        {
            "id": 2,
            "complaint_text": "Hostel Wi-Fi keeps disconnecting and blocks internet access.",
            "status": "In Progress",
        },
    ]

    matches = find_similar_submitted_complaints(
        "The hostel Wi-Fi disconnects and students cannot access internet.",
        complaints,
        build_vectorizer(),
    )

    assert len(matches) == 1
    assert matches[0]["id"] == 2
    assert matches[0]["status"] == "In Progress"
    assert matches[0]["similarity"] >= 25


def test_does_not_return_unrelated_complaints():
    complaints = [{
        "id": 1,
        "complaint_text": "The cafeteria meals are cold and food quality is poor.",
        "status": "Pending",
    }]

    matches = find_similar_submitted_complaints(
        "The hostel Wi-Fi disconnects and internet access is unavailable.",
        complaints,
        build_vectorizer(),
    )

    assert matches == []


def test_returns_no_matches_when_no_complaints_exist():
    assert find_similar_submitted_complaints(
        "The hostel Wi-Fi is not working.",
        [],
        build_vectorizer(),
    ) == []


def test_skips_blank_complaints():
    complaints = [
        {"id": 1, "complaint_text": "  ", "status": "Pending"},
        {
            "id": 2,
            "complaint_text": "Hostel Wi-Fi keeps disconnecting and blocks internet access.",
            "status": "Pending",
        },
    ]

    matches = find_similar_submitted_complaints(
        "The hostel Wi-Fi disconnects and students cannot access internet.",
        complaints,
        build_vectorizer(),
    )

    assert [match["id"] for match in matches] == [2]


def test_rejects_invalid_similarity_threshold():
    with pytest.raises(ValueError, match="min_similarity"):
        find_similar_submitted_complaints(
            "The hostel Wi-Fi is not working.",
            [],
            build_vectorizer(),
            min_similarity=101,
        )
