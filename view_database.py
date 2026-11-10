import sqlite3
import pandas as pd

connection = sqlite3.connect("campuspulse.db")

df = pd.read_sql_query(
    "SELECT id, complaint_text, department_predictions, urgency, sentiment, priority, status FROM complaints",
    connection
)

print(df.to_string(index=False))

connection.close()