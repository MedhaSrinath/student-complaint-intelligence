import sqlite3

import complaint_database
from complaint_database import (
    add_complaint_support,
    create_database,
    get_all_complaints,
)


def test_add_complaint_support_increments_count(tmp_path, monkeypatch):
    database_path = tmp_path / "campuspulse.db"
    monkeypatch.setattr(complaint_database, "DATABASE", str(database_path))
    create_database()

    connection = sqlite3.connect(database_path)
    connection.execute(
        "INSERT INTO complaints (complaint_text, submitted_at) "
        "VALUES (?, ?)",
        ("Hostel Wi-Fi is down.", "2026-01-01 10:00:00"),
    )
    connection.commit()
    connection.close()

    assert add_complaint_support(1) is True
    assert add_complaint_support(1) is True
    assert get_all_complaints()[0]["support_count"] == 2
    assert add_complaint_support(999) is False


def test_create_database_adds_support_count_to_existing_table(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "legacy.db"
    monkeypatch.setattr(complaint_database, "DATABASE", str(database_path))
    connection = sqlite3.connect(database_path)
    connection.execute("""
        CREATE TABLE complaints (
            id INTEGER PRIMARY KEY,
            complaint_text TEXT NOT NULL,
            submitted_at TEXT NOT NULL
        )
    """)
    connection.execute(
        "INSERT INTO complaints (id, complaint_text, submitted_at) "
        "VALUES (1, ?, ?)",
        ("The hostel Wi-Fi is down.", "2026-01-01 10:00:00"),
    )
    connection.commit()
    connection.close()

    create_database()

    complaint = get_all_complaints()[0]
    assert complaint["support_count"] == 0
