"""
app.py
------
Main Flask application for Skin Insight.

Routes
------
POST /register    – Register a new user (name, email).
POST /upload      – Upload a skin image; stores the file and DB record.
POST /predict     – Run the CNN model on an uploaded image; stores prediction.
GET  /recommend   – Fetch product recommendations for the latest prediction.

Run the server
--------------
    python backend/app.py
"""

import os
import sqlite3
import uuid
import csv
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.preprocess_utils import preprocess_image
from utils.model_utils import predict

# ── Application Setup ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

# Directory where uploaded images are stored (relative to project root)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Path to the SQLite database
DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")

# Path to the products CSV used for recommendations
PRODUCTS_CSV = os.path.join(os.path.dirname(__file__), "..", "datasets", "products.csv")


# ── Database Helper ────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """Open and return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Rows accessible as dicts
    return conn


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/register", methods=["POST"])
def register():
    """
    Register a new user.

    Expected JSON body
    ------------------
    { "name": "Alice", "email": "alice@example.com" }

    Returns
    -------
    JSON with the new user's id, or an error message.
    """
    data = request.get_json()
    if not data or not data.get("name") or not data.get("email"):
        return jsonify({"error": "Name and email are required."}), 400

    name = data["name"].strip()
    email = data["email"].strip().lower()

    try:
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)", (name, email)
            )
            conn.commit()
            user_id = cursor.lastrowid

        return jsonify({"message": "User registered successfully.", "user_id": user_id}), 201

    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered."}), 409

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload a skin image.

    Expected multipart/form-data fields
    ------------------------------------
    file    – The image file (JPEG, PNG, etc.).
    user_id – (optional) The ID of the registered user.

    Returns
    -------
    JSON with the saved image_id and image_path.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    user_id = request.form.get("user_id")

    # Generate a unique filename to avoid collisions
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(IMAGES_DIR, filename)
    file.save(save_path)

    # Store relative path in DB for portability
    relative_path = os.path.join("images", filename)

    try:
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO images (user_id, image_path) VALUES (?, ?)",
                (user_id, relative_path),
            )
            conn.commit()
            image_id = cursor.lastrowid

        return jsonify({
            "message": "Image uploaded successfully.",
            "image_id": image_id,
            "image_path": relative_path,
        }), 201

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Predict the skin condition from an uploaded image.

    Expected multipart/form-data fields
    ------------------------------------
    file – The image file to analyse.

    OR JSON body with
    ------------------------------------
    image_id – ID of a previously uploaded image (uses stored file).

    Returns
    -------
    JSON with predicted_class, confidence, and image_id.
    """
    image_id = None

    # ── Option 1: raw file upload (upload + predict in one step) ────────────
    if "file" in request.files:
        file = request.files["file"]
        file_bytes = file.read()
        user_id = request.form.get("user_id")

        # Save the file
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(IMAGES_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        relative_path = os.path.join("images", filename)

        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO images (user_id, image_path) VALUES (?, ?)",
                (user_id, relative_path),
            )
            conn.commit()
            image_id = cursor.lastrowid

    # ── Option 2: use a previously uploaded image ───────────────────────────
    elif request.is_json:
        data = request.get_json()
        image_id = data.get("image_id")
        if not image_id:
            return jsonify({"error": "Provide either a file or an image_id."}), 400

        with get_db() as conn:
            row = conn.execute(
                "SELECT image_path FROM images WHERE id = ?", (image_id,)
            ).fetchone()

        if not row:
            return jsonify({"error": "Image not found."}), 404

        image_path = os.path.join(os.path.dirname(__file__), "..", row["image_path"])
        with open(image_path, "rb") as f:
            file_bytes = f.read()

    else:
        return jsonify({"error": "Provide either a file upload or a JSON body with image_id."}), 400

    try:
        # Preprocess the image into a numpy array
        image_array = preprocess_image(file_bytes)

        # Run model inference
        predicted_class, confidence = predict(image_array)

        # Store the prediction in the database
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO predictions (image_id, predicted_class, confidence)
                VALUES (?, ?, ?)
                """,
                (image_id, predicted_class, confidence),
            )
            conn.commit()

        return jsonify({
            "image_id": image_id,
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
        }), 200

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Return product recommendations for a predicted skin condition.

    Query Parameters
    ----------------
    skin_problem : str
        One of: Acne, Blackheads, Dark Spots, Normal, Pores, Wrinkles.
    image_id : int (optional)
        If provided, saves the recommendations to the DB.

    Returns
    -------
    JSON with a list of recommended products and the total price.
    """
    skin_problem = request.args.get("skin_problem", "").strip()
    image_id = request.args.get("image_id")

    if not skin_problem:
        return jsonify({"error": "skin_problem query parameter is required."}), 400

    # Read products from CSV
    if not os.path.exists(PRODUCTS_CSV):
        return jsonify({"error": "Products database not found."}), 503

    recommended = []
    with open(PRODUCTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["associated_skin_problem"].strip().lower() == skin_problem.lower():
                recommended.append({
                    "product_name": row["product_name"],
                    "brand": row["brand"],
                    "price": float(row["price"]),
                    "associated_skin_problem": row["associated_skin_problem"],
                })

    total_price = sum(p["price"] for p in recommended)

    # Optionally persist recommendations in the DB
    if image_id:
        try:
            with get_db() as conn:
                for product in recommended:
                    # Look up the product id
                    row = conn.execute(
                        "SELECT id FROM products WHERE product_name = ? AND brand = ?",
                        (product["product_name"], product["brand"]),
                    ).fetchone()
                    if row:
                        conn.execute(
                            "INSERT INTO recommendations (image_id, product_id) VALUES (?, ?)",
                            (image_id, row["id"]),
                        )
                conn.commit()
        except Exception:
            pass  # Recommendations are still returned even if DB write fails

    return jsonify({
        "skin_problem": skin_problem,
        "products": recommended,
        "total_price": round(total_price, 2),
        "currency": "Rs",
    }), 200


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure DB exists before handling any requests
    if not os.path.exists(DB_PATH):
        print("Database not found. Run: python backend/create_db.py")

    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
