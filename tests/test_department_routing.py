import pytest

from department_routing import (
    maintenance_routing_reason,
    rank_department_predictions,
)


@pytest.mark.parametrize(
    "text",
    [
        "lift in f block is not working",
        "The elevator is stuck between floors.",
        "The lift is broken; please send a technician.",
        "The elevator is out of service.",
    ],
)
def test_routes_lift_faults_to_maintenance(text):
    assert maintenance_routing_reason(text) == (
        "Lift/elevator fault or repair request"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Please add a mirror in the elevator.",
        "The lift is crowded between classes.",
        "The hostel Wi-Fi is not working.",
    ],
)
def test_does_not_route_non_repair_requests_to_maintenance(text):
    assert maintenance_routing_reason(text) is None


def test_maintenance_rule_can_promote_a_department_outside_model_top_three():
    predictions = [
        {"department": "Hostels", "confidence": 28.5},
        {"department": "Academics", "confidence": 13.1},
        {"department": "IT Support", "confidence": 12.8},
        {"department": "Maintenance", "confidence": 8.2},
        {"department": "Transport", "confidence": 7.0},
    ]

    result = rank_department_predictions(
        "lift in f block is not working",
        predictions,
    )

    assert result[0] == {
        "department": "Maintenance",
        "confidence": 8.2,
        "routing_reason": "Lift/elevator fault or repair request",
    }
    assert len(result) == 3


def test_non_maintenance_complaints_keep_model_ranking():
    predictions = [
        {"department": "Hostels", "confidence": 40.0},
        {"department": "Maintenance", "confidence": 30.0},
        {"department": "Academics", "confidence": 20.0},
        {"department": "IT Support", "confidence": 10.0},
    ]

    assert rank_department_predictions(
        "The hostel Wi-Fi is not working.",
        predictions,
    ) == predictions[:3]
