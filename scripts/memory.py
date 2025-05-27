import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import sqlite3
from datetime import datetime

# Connect to SQLite database (will create if it doesn't exist)
conn = sqlite3.connect("memory.db")
cursor = conn.cursor()

# Create memory table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def save_memory(prompt: str, response: str):
    """Save a prompt-response pair to the memory database."""
    cursor.execute(
        "INSERT INTO memory (prompt, response) VALUES (?, ?)",
        (prompt, response)
    )
    conn.commit()

def recall_memories(limit: int = 5):
    """Retrieve the most recent prompt-response pairs."""
    cursor.execute(
        "SELECT prompt, response, timestamp FROM memory ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    return cursor.fetchall()

def clear_memory():
    """Delete all memory records."""
    cursor.execute("DELETE FROM memory")
    conn.commit()
