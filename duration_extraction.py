import re


_NUMBER = (
    r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|a|an|half\s+an?|couple\s+of|few|several)"
)
_UNIT = r"(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)"
_DURATION_PATTERN = re.compile(
    rf"\b(?:(?:for|over|during|lasting|lasted)\s+"
    rf"(?:(?:the\s+)?(?:past|last|about|approximately|around|nearly|almost)\s+)*)?"
    rf"{_NUMBER}\s+{_UNIT}\b",
    re.IGNORECASE,
)

_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_MONTH = (
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)"
)
_DATE = (
    rf"(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}|"
    rf"{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH})"
)
_TIME_REFERENCE_PATTERN = re.compile(
    rf"\bsince\s+(?:(?:(?:last|this)\s+)?"
    rf"(?:{_WEEKDAY}|week|month|year)|yesterday|today|"
    rf"(?:this|last)\s+(?:morning|afternoon|evening)|{_DATE})\b",
    re.IGNORECASE,
)


def extract_duration_mentions(text: str) -> list[dict[str, str]]:
    mentions = [
        {
            "entity": match.group().strip(),
            "type": "DURATION",
            "description": "Reported duration or start-time reference",
        }
        for pattern in (_DURATION_PATTERN, _TIME_REFERENCE_PATTERN)
        for match in pattern.finditer(text)
    ]

    mentions.sort(key=lambda mention: text.lower().find(mention["entity"].lower()))
    return mentions
