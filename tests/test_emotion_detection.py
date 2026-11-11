import pytest

from emotion_detection import detect_emotion


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I am furious about the broken lab equipment.", "Anger"),
        ("I am fed up with the repeated Wi-Fi failures.", "Frustration"),
        ("I feel disappointed that the issue remains unresolved.", "Sadness"),
        ("I am worried about the exposed wire.", "Fear"),
        ("The library printer is not working.", "Neutral"),
    ],
)
def test_detects_emotion_from_clear_wording(text, expected):
    assert detect_emotion(text)["emotion"] == expected


def test_returns_matching_cues_for_explanation():
    result = detect_emotion("I am frustrated and annoyed by the delay.")

    assert result == {
        "emotion": "Frustration",
        "cues": ["frustrated", "annoyed"],
    }


@pytest.mark.parametrize(
    "text",
    [
        "I am not angry about the schedule.",
        "I am not worried about the delay.",
    ],
)
def test_ignores_emotion_cues_preceded_by_negation(text):
    assert detect_emotion(text) == {"emotion": "Neutral", "cues": []}
