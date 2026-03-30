"""
create_db.py
------------
Creates the SQLite database and all required tables for Skin Insight.
Also inserts sample product data for testing.

Run once before starting the Flask server:
    python backend/create_db.py
"""

import sqlite3
import os

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")

# Path to the products CSV (used to seed the products table)
PRODUCTS_CSV = os.path.join(os.path.dirname(__file__), "..", "datasets", "products.csv")


def get_connection():
    """Return a new SQLite connection to the application database."""
    return sqlite3.connect(DB_PATH)


def create_tables(conn: sqlite3.Connection) -> None:
    """
    Create all required tables if they do not already exist.

    Tables
    ------
    users          – Registered application users.
    images         – Images uploaded by users.
    predictions    – ML model predictions for uploaded images.
    products       – Skincare product catalogue.
    recommendations – Product recommendations linked to a prediction.
    """
    cursor = conn.cursor()

    # ── users ────────────────────────────────────────────────────────────────
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            email      TEXT    NOT NULL UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # ── images ───────────────────────────────────────────────────────────────
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
            image_path  TEXT    NOT NULL,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # ── predictions ──────────────────────────────────────────────────────────
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            predicted_class TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            predicted_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # ── products ─────────────────────────────────────────────────────────────
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name           TEXT NOT NULL,
            brand                  TEXT NOT NULL,
            price                  REAL NOT NULL,
            associated_skin_problem TEXT NOT NULL
        )
        """
    )

    # ── recommendations ───────────────────────────────────────────────────────
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE
        )
        """
    )

    conn.commit()
    print("All tables created (or already exist).")


def seed_products(conn: sqlite3.Connection) -> None:
    """
    Insert sample products from products.csv into the products table.
    Skips seeding if rows already exist to avoid duplicates on re-runs.
    """
    cursor = conn.cursor()

    # Check if products table already has data
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    if count > 0:
        print(f"Products table already has {count} rows — skipping seed.")
        return

    import csv

    if not os.path.exists(PRODUCTS_CSV):
        print(f"Warning: products.csv not found at {PRODUCTS_CSV}. Skipping seed.")
        return

    with open(PRODUCTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            (row["product_name"], row["brand"], float(row["price"]), row["associated_skin_problem"])
            for row in reader
        ]

    cursor.executemany(
        "INSERT INTO products (product_name, brand, price, associated_skin_problem) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"Inserted {len(rows)} products into the database.")


if __name__ == "__main__":
    with get_connection() as conn:
        create_tables(conn)
        seed_products(conn)
    print(f"Database ready at: {DB_PATH}")
