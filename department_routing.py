import re

# Simple mapping from department label to email address (temporary testing address).
# Update the email values as needed for real routing.
_DEPARTMENT_EMAIL_MAP = {
    "Academic": "medhasrinathbsc24@rvu.edu.in",
    "Administrative": "medhasrinathbsc24@rvu.edu.in",
    "Finance": "medhasrinathbsc24@rvu.edu.in",
    "Infrastructure": "medhasrinathbsc24@rvu.edu.in",
    "Technical": "medhasrinathbsc24@rvu.edu.in",
    "Maintenance": "medhasrinathbsc24@rvu.edu.in",
}


def get_department_email(department: str) -> str | None:
    return _DEPARTMENT_EMAIL_MAP.get(department)


_LIFT_OR_ELEVATOR = re.compile(r"\b(?:lift|elevators?)\b", re.IGNORECASE)

_MAINTENANCE_CUE = re.compile(
    r"\b(?:"
    r"broken|damaged|faulty|failed|failure|"
    r"malfunction(?:ing)?|stuck|jammed|"
    r"out\s+of\s+service|not\s+working|"
    r"does(?:n't|\s+not)\s+work|"
    r"won't\s+work|won't\s+move|"
    r"stop(?:ped)?\s+working|repair|"
    r"maintenance|technician"
    r")\b",
    re.IGNORECASE,
)


def maintenance_routing_reason(text: str) -> str | None:
    if _LIFT_OR_ELEVATOR.search(text) and _MAINTENANCE_CUE.search(text):
        return "Lift/elevator fault or repair request"

    return None


def rank_department_predictions(text, predictions):
    ranked = [dict(prediction) for prediction in predictions]
    reason = maintenance_routing_reason(text)

    if reason:
        maintenance = next(
            (
                prediction
                for prediction in ranked
                if prediction["department"] == "Maintenance"
            ),
            None,
        )

        if maintenance is not None:
            maintenance["routing_reason"] = reason

            others = [
                prediction
                for prediction in ranked
                if prediction["department"] != "Maintenance"
            ]

            return [maintenance, *others[:2]]

    return ranked[:3]