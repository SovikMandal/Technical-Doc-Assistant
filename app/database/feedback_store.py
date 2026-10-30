import sqlite3
from pathlib import Path

# Store the database in the project root
DB_PATH = Path(__file__).resolve().parents[2] / "feedback.db"


def initialize_feedback_db():
    """Create the feedback table if it doesn't exist."""

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()


def save_feedback(
    question: str,
    answer: str,
    rating: int,
    feedback: str | None = None
) -> int:
    """Save user feedback and return its database ID."""

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO feedback (question, answer, rating, feedback)
            VALUES (?, ?, ?, ?)
            """,
            (question, answer, rating, feedback)
        )

        connection.commit()

        return cursor.lastrowid


def get_all_feedback():
    """Retrieve all saved feedback."""

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT * FROM feedback ORDER BY created_at DESC
        """).fetchall()

        return [dict(row) for row in rows]
