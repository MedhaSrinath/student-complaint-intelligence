import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from duplicate_detection import (
    find_recurring_complaint_groups,
    find_similar_submitted_complaints,
)
from evaluate_duplicate_detection import evaluate_thresholds


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


def test_uses_the_provisionally_evaluated_similarity_cutoff():
    vectorizer = build_vectorizer()
    existing = [{
        "id": 1,
        "complaint_text": "The hostel Wi-Fi keeps disconnecting and students cannot access internet.",
        "status": "Pending",
    }]
    query = "Wireless network problems for students."

    match = find_similar_submitted_complaints(
        query,
        existing,
        vectorizer,
    )
    rejected = find_similar_submitted_complaints(
        query,
        existing,
        vectorizer,
        min_similarity=26,
    )

    assert len(match) == 1
    assert 20 <= match[0]["similarity"] < 26
    assert rejected == []


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


def test_groups_similar_complaints_and_keeps_unrelated_reports_separate():
    complaints = [
        {
            "id": 3,
            "complaint_text": "The cafeteria meals are cold and food quality is poor.",
            "status": "Pending",
        },
        {
            "id": 2,
            "complaint_text": "Hostel Wi-Fi keeps disconnecting and blocks internet access.",
            "status": "Resolved",
        },
        {
            "id": 1,
            "complaint_text": "The hostel Wi-Fi disconnects and students cannot access internet.",
            "status": "Pending",
        },
    ]

    groups = find_recurring_complaint_groups(complaints, build_vectorizer())

    related_group = next(group for group in groups if group["count"] == 2)
    individual_group = next(group for group in groups if group["count"] == 1)

    assert related_group["topic_label"] == "Hostel Wi-Fi"
    assert {
        member["id"] for member in related_group["complaints"]
    } == {1, 2}
    assert individual_group["complaints"][0]["id"] == 3


def test_keeps_individual_issues_available_for_support():
    complaints = [
        {
            "id": 1,
            "complaint_text": "The cafeteria meals are cold and food quality is poor.",
            "status": "Pending",
        },
        {
            "id": 2,
            "complaint_text": "Hostel Wi-Fi keeps disconnecting and blocks internet access.",
            "status": "Pending",
        },
    ]

    groups = find_recurring_complaint_groups(complaints, build_vectorizer())

    assert len(groups) == 2
    assert all(group["count"] == 1 for group in groups)
    assert {group["complaints"][0]["id"] for group in groups} == {1, 2}


def test_single_submitted_complaint_can_receive_support():
    complaints = [{
        "id": 1,
        "complaint_text": "wi-fi is not working",
        "status": "Pending",
        "support_count": 0,
    }]

    groups = find_recurring_complaint_groups(complaints, build_vectorizer())

    assert len(groups) == 1
    assert groups[0]["count"] == 1
    assert groups[0]["topic_label"] == "Wi-Fi"


def test_support_votes_add_to_group_count_and_can_create_a_group():
    complaints = [{
        "id": 1,
        "complaint_text": "The hostel Wi-Fi keeps disconnecting and students cannot access internet.",
        "status": "Pending",
        "support_count": 2,
    }]

    groups = find_recurring_complaint_groups(complaints, build_vectorizer())

    assert len(groups) == 1
    assert groups[0]["count"] == 3
    assert groups[0]["complaints"][0]["support_count"] == 2


def test_evaluates_threshold_precision_recall_and_f1():
    result = evaluate_thresholds(
        [True, True, False, False],
        [0.8, 0.3, 0.7, 0.1],
        [50],
    )[0]

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_rejects_mismatched_evaluation_inputs():
    with pytest.raises(ValueError, match="same length"):
        evaluate_thresholds([True], [], [25])
