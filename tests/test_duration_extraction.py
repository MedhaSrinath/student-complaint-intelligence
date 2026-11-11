import pytest

from duration_extraction import extract_duration_mentions
from entity_extraction import extract_entities


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The leak has continued for three days.", ["for three days"]),
        ("Wi-Fi has been down for the past 2 weeks.", ["for the past 2 weeks"]),
        ("It stopped working about 30 minutes ago.", ["30 minutes"]),
        ("The issue has persisted since last Monday.", ["since last Monday"]),
        ("The issue has persisted since this morning.", ["since this morning"]),
        ("The portal has been unavailable since March 4, 2025.", [
            "since March 4, 2025",
        ]),
        ("Repairs started on Monday.", []),
        ("Please help the students in the library.", []),
    ],
)
def test_extracts_duration_and_start_time_mentions(text, expected):
    mentions = extract_duration_mentions(text)

    assert [mention["entity"] for mention in mentions] == expected
    assert all(mention["type"] == "DURATION" for mention in mentions)


def test_returns_multiple_mentions_in_text_order():
    mentions = extract_duration_mentions(
        "The leak began since Monday and has continued for two days."
    )

    assert [mention["entity"] for mention in mentions] == [
        "since Monday",
        "for two days",
    ]


def test_duration_mentions_are_included_with_other_entities():
    entities = extract_entities(
        "Water is leaking in Block A for three days."
    )

    assert {
        (entity["entity"], entity["type"])
        for entity in entities
    } >= {
        ("Block A", "CAMPUS_LOCATION"),
        ("for three days", "DURATION"),
    }
