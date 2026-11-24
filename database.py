
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("prediction_history.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time TEXT NOT NULL,
                stock_ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                closing_price REAL,
                predicted_trend TEXT NOT NULL,
                confidence REAL,
                prediction_type TEXT NOT NULL
            )
        """)


def save_prediction(
    ticker,
    prediction_date,
    closing_price,
    predicted_trend,
    confidence,
    prediction_type
):
    create_table()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions (
                prediction_time,
                stock_ticker,
                prediction_date,
                closing_price,
                predicted_trend,
                confidence,
                prediction_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker,
            str(prediction_date),
            float(closing_price),
            predicted_trend,
            float(confidence),
            prediction_type
        ))


def get_history():
    create_table()

    with get_connection() as conn:
        return conn.execute("""
            SELECT * FROM predictions
            ORDER BY id DESC
        """).fetchall()


def clear_history():
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions")
