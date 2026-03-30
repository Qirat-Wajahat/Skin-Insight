"""
test_pipeline.py
----------------
Simulates the full Skin Insight pipeline without requiring a trained model:

    1. Database creation (tables + product seed).
    2. User registration via /register.
    3. Image upload via /upload.
    4. Product recommendation via /recommend.
    5. SQLite table verification.

Note: /predict requires a trained model (backend/models/skin_model.h5).
      The test mocks model inference so the full pipeline can be verified
      without a trained model.

Run from the project root:
    python scripts/test_pipeline.py
"""

import os
import sys
import json
import sqlite3
import tempfile
import io
import unittest
from unittest.mock import patch, MagicMock

# ── Ensure the backend package is importable ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

TEST_PRODUCTS_CSV = os.path.join(PROJECT_ROOT, "datasets", "products.csv")


class TestSkinInsightPipeline(unittest.TestCase):
    """End-to-end pipeline tests for the Skin Insight Flask API."""

    @classmethod
    def setUpClass(cls):
        """Initialise a shared test client with a temp-file DB (not :memory:)."""
        import backend.app as app_module
        from backend.create_db import create_tables, seed_products

        # Use a real temp file so all connections share the same DB
        cls._tmp_dir  = tempfile.TemporaryDirectory()
        cls._db_file  = os.path.join(cls._tmp_dir.name, "test.db")

        app_module.IMAGES_DIR = cls._tmp_dir.name
        app_module.DB_PATH    = cls._db_file

        with sqlite3.connect(cls._db_file) as conn:
            create_tables(conn)
            if os.path.exists(TEST_PRODUCTS_CSV):
                seed_products(conn)

        app_module.app.config["TESTING"] = True
        cls.client     = app_module.app.test_client()
        cls.app_module = app_module

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()

    # ── Test 1: Database Tables Exist ────────────────────────────────────────
    def test_01_database_tables(self):
        """All required tables should be present after create_db runs."""
        with sqlite3.connect(self._db_file) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        expected = {"users", "images", "predictions", "products", "recommendations"}
        self.assertTrue(expected.issubset(tables), f"Missing tables: {expected - tables}")
        print("  ✓ All database tables exist.")

    # ── Test 2: Products Seeded ───────────────────────────────────────────────
    def test_02_products_seeded(self):
        """The products table should have at least one row."""
        with sqlite3.connect(self._db_file) as conn:
            count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        self.assertGreater(count, 0, "Products table is empty – seed may have failed.")
        print(f"  ✓ {count} products seeded.")

    # ── Test 3: User Registration ─────────────────────────────────────────────
    def test_03_register_user(self):
        """POST /register should create a user and return a user_id."""
        res = self.client.post(
            "/register",
            data=json.dumps({"name": "Test User", "email": "test@skininsight.com"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertIn("user_id", data)
        self.__class__.user_id = data["user_id"]
        print(f"  ✓ User registered with id={data['user_id']}.")

    # ── Test 4: Duplicate Email Rejected ──────────────────────────────────────
    def test_04_duplicate_email(self):
        """POST /register with a duplicate email should return 409."""
        res = self.client.post(
            "/register",
            data=json.dumps({"name": "Test User 2", "email": "test@skininsight.com"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 409)
        print("  ✓ Duplicate email correctly rejected.")

    # ── Test 5: Image Upload ──────────────────────────────────────────────────
    def test_05_upload_image(self):
        """POST /upload should store the image and return an image_id."""
        # Create a minimal 1×1 white JPEG in memory
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (10, 10), color=(255, 200, 180)).save(buf, format="JPEG")
        buf.seek(0)

        res = self.client.post(
            "/upload",
            data={"file": (buf, "test_skin.jpg"), "user_id": getattr(self, "user_id", "")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertIn("image_id", data)
        self.__class__.image_id = data["image_id"]
        print(f"  ✓ Image uploaded with id={data['image_id']}.")

    # ── Test 6: Predict (mocked model) ────────────────────────────────────────
    def test_06_predict(self):
        """POST /predict should return predicted_class and confidence."""
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (224, 224), color=(220, 180, 160)).save(buf, format="JPEG")
        buf.seek(0)

        # Patch the `predict` name as it is referenced inside backend/app.py
        with patch.object(self.app_module, "predict", return_value=("Acne", 0.87)):
            res = self.client.post(
                "/predict",
                data={"file": (buf, "face.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("predicted_class", data)
        self.assertIn("confidence", data)
        self.__class__.predicted_class = data["predicted_class"]
        print(f"  ✓ Prediction: {data['predicted_class']} ({data['confidence']}% confidence).")

    # ── Test 7: Product Recommendations ──────────────────────────────────────
    def test_07_recommend(self):
        """GET /recommend?skin_problem=Acne should return relevant products."""
        res = self.client.get("/recommend?skin_problem=Acne")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("products", data)
        self.assertGreater(len(data["products"]), 0)
        self.assertIn("total_price", data)
        print(
            f"  ✓ {len(data['products'])} products recommended for Acne. "
            f"Total: Rs {data['total_price']}"
        )

    # ── Test 8: Recommend Missing Parameter ──────────────────────────────────
    def test_08_recommend_missing_param(self):
        """GET /recommend without skin_problem should return 400."""
        res = self.client.get("/recommend")
        self.assertEqual(res.status_code, 400)
        print("  ✓ Missing skin_problem parameter correctly rejected.")

    # ── Test 9: Register Validation ───────────────────────────────────────────
    def test_09_register_validation(self):
        """POST /register without email should return 400."""
        res = self.client.post(
            "/register",
            data=json.dumps({"name": "No Email"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        print("  ✓ Registration validation works correctly.")


if __name__ == "__main__":
    print("\n╔══════════════════════════════════╗")
    print("║  Skin Insight – Pipeline Tests   ║")
    print("╚══════════════════════════════════╝\n")

    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = lambda a, b: (a > b) - (a < b)  # alphabetical = numeric order
    suite  = loader.loadTestsFromTestCase(TestSkinInsightPipeline)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ All tests passed.")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
