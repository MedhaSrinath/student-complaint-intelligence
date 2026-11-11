import re


_EMOTION_CUES = {
    "Anger": ("angry", "furious", "outraged", "infuriated", "hate"),
    "Frustration": (
        "frustrated",
        "frustrating",
        "fed up",
        "annoyed",
        "irritated",
        "exasperated",
    ),
    "Sadness": (
        "sad",
        "disappointed",
        "unhappy",
        "miserable",
        "hopeless",
        "let down",
    ),
    "Fear": (
        "afraid",
        "scared",
        "worried",
        "anxious",
        "terrified",
        "unsafe",
    ),
}
_NEGATIONS = {"no", "not", "never", "without", "cannot", "can't", "dont", "don't"}


def detect_emotion(text: str) -> dict[str, str | list[str]]:
    normalized_text = text.lower()
    matches = {}

    for emotion, cues in _EMOTION_CUES.items():
        found = []
        for cue in cues:
            pattern = rf"(?<!\w){re.escape(cue)}(?!\w)"
            for match in re.finditer(pattern, normalized_text):
                preceding_words = re.findall(
                    r"\b[\w']+\b",
                    normalized_text[:match.start()],
                )[-3:]
                if not _NEGATIONS.intersection(preceding_words):
                    found.append(match.group())
        if found:
            matches[emotion] = list(dict.fromkeys(found))

    if not matches:
        return {"emotion": "Neutral", "cues": []}

    emotion = max(
        matches,
        key=lambda label: (
            len(matches[label]),
            -list(_EMOTION_CUES).index(label),
        ),
    )
    return {"emotion": emotion, "cues": matches[emotion]}
