import sqlite3
import json
from datetime import datetime

DATABASE = "campuspulse.db"

def create_database():
    """Create the complaints table if it does not already exist.
    Preserves existing data; uses IF NOT EXISTS so no records are lost.
    The schema matches the original columns used throughout the app.
    """
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_text TEXT NOT NULL,
            submitted_at TEXT NOT NULL,
            department_predictions TEXT,
            urgency TEXT,
            urgency_confidence REAL,
            sentiment TEXT,
            sentiment_confidence REAL,
            emotion TEXT,
            priority_score REAL,
            priority TEXT,
            keywords TEXT,
            entities TEXT,
            semantic_matches TEXT,
            tfidf_matches TEXT,
            status TEXT DEFAULT 'Pending',
            support_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    connection.commit()
    connection.close()

def save_complaint(text, result):
    """Insert a new complaint into the database and return its ID."""
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO complaints (
            complaint_text,
            submitted_at,
            department_predictions,
            urgency,
            urgency_confidence,
            sentiment,
            sentiment_confidence,
            emotion,
            priority_score,
            priority,
            keywords,
            entities,
            semantic_matches,
            tfidf_matches
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        text,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        json.dumps(result["department_predictions"]),
        result["urgency"]["urgency"],
        float(result["urgency"]["confidence"]),
        result["sentiment"]["sentiment"],
        float(result["sentiment"]["confidence"]),
        result.get("emotion", {}).get("emotion", "Neutral"),
        float(result["priority"]["score"]),
        result["priority"]["priority"],
        json.dumps(result["keywords"]),
        json.dumps(result["entities"]),
        json.dumps(result["semantic_similar_complaints"]),
        json.dumps(result["tfidf_similar_complaints"])
    ))
    connection.commit()
    complaint_id = cursor.lastrowid
    connection.close()
    return complaint_id

def get_all_complaints():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM complaints ORDER BY id DESC")
    complaints = [dict(row) for row in cursor.fetchall()]
    connection.close()
    return complaints

def update_complaint_status(complaint_id, status):
    allowed_statuses = ["Pending", "In Progress", "Resolved"]
    if status not in allowed_statuses:
        raise ValueError("Invalid complaint status")
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE complaints SET status = ? WHERE id = ?",
        (status, complaint_id)
    )
    connection.commit()
    updated = cursor.rowcount
    connection.close()
    return updated > 0

def add_complaint_support(complaint_id):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE complaints SET support_count = support_count + 1 WHERE id = ?",
        (complaint_id,)
    )
    connection.commit()
    updated = cursor.rowcount
    connection.close()
    return updated > 0

if __name__ == "__main__":
    create_database()
