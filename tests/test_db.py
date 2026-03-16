"""
Tests for the PostgreSQL database module.

Run with: python -m pytest tests/test_db.py -v

These tests exercise the Database class against a real (temporary) PostgreSQL
database spun up by the test session.  If PostgreSQL is not running locally,
the tests are automatically *skipped* — not failed — so CI and local dev
without Docker still get a green suite.

A second group of "unit" tests verifies the object interface (method
signatures, degradation behaviour) without needing a live database.
"""

import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

# Allow importing from services/inference/
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "services", "inference")
)

from db import Database

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "localization")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "changeme123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "pet_tracking")


def _can_connect() -> bool:
    """Quick probe to see if Postgres is reachable."""
    d = Database(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )
    ok = d.connect()
    d.close()
    return ok


_pg_available = _can_connect()
requires_postgres = pytest.mark.skipif(
    not _pg_available, reason="PostgreSQL not available"
)


# ============================================================================
# Unit tests (no database required)
# ============================================================================

class TestDatabaseUnit:
    """Verify interface and degradation without a live database."""

    def test_initial_state(self):
        db = Database(host="nowhere", port=9999)
        assert not db.connected

    def test_write_position_returns_false_when_disconnected(self):
        db = Database(host="nowhere", port=9999)
        # Override retry interval so connect attempt happens immediately
        import db as db_mod
        original = db_mod._RETRY_INTERVAL
        db_mod._RETRY_INTERVAL = 0.0
        try:
            result = db.write_position(
                beacon_id="test",
                x=1.0, y=2.0, floor=1,
                confidence=0.5,
                location_label="office",
            )
            assert result is False
        finally:
            db_mod._RETRY_INTERVAL = original

    def test_write_fingerprint_returns_false_when_disconnected(self):
        db = Database(host="nowhere", port=9999)
        import db as db_mod
        original = db_mod._RETRY_INTERVAL
        db_mod._RETRY_INTERVAL = 0.0
        try:
            result = db.write_fingerprint(
                location_label="kitchen",
                floor=2,
                grid_x=5.0, grid_y=10.0,
                rssi_vector={"a": -65},
            )
            assert result is False
        finally:
            db_mod._RETRY_INTERVAL = original

    def test_read_position_returns_empty_when_disconnected(self):
        db = Database(host="nowhere", port=9999)
        import db as db_mod
        original = db_mod._RETRY_INTERVAL
        db_mod._RETRY_INTERVAL = 0.0
        try:
            result = db.read_position_history()
            assert result == []
        finally:
            db_mod._RETRY_INTERVAL = original

    def test_count_returns_zero_when_disconnected(self):
        db = Database(host="nowhere", port=9999)
        import db as db_mod
        original = db_mod._RETRY_INTERVAL
        db_mod._RETRY_INTERVAL = 0.0
        try:
            assert db.count_positions() == 0
            assert db.count_fingerprints() == 0
        finally:
            db_mod._RETRY_INTERVAL = original

    def test_retry_backoff_skips_rapid_reconnects(self):
        """Verify that rapid connect() calls are silently skipped."""
        db = Database(host="nowhere", port=9999)
        import db as db_mod
        original = db_mod._RETRY_INTERVAL
        db_mod._RETRY_INTERVAL = 999.0  # very long
        try:
            # First attempt resets the timer
            db._last_attempt = 0.0
            db.connect()
            # Second attempt within the interval should be a no-op
            db._last_attempt = time.time()
            result = db.connect()
            assert result is False  # still disconnected, but didn't actually try
        finally:
            db_mod._RETRY_INTERVAL = original

    def test_close_idempotent(self):
        db = Database(host="nowhere", port=9999)
        db.close()  # should not raise
        db.close()


# ============================================================================
# Integration tests (require running PostgreSQL)
# ============================================================================

@requires_postgres
class TestDatabaseIntegration:
    """Full round-trip tests against a live PostgreSQL instance."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Provide a connected Database and clean up test data after."""
        self.db = Database(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DB,
        )
        assert self.db.connect()
        yield
        # Clean up test rows (our test beacon id is unlikely to collide)
        from sqlalchemy import text
        with self.db._engine.connect() as conn:
            conn.execute(
                text("DELETE FROM position_history WHERE beacon_id LIKE 'test_%'")
            )
            conn.execute(
                text("DELETE FROM fingerprint_samples WHERE notes LIKE 'test_%'")
            )
            conn.commit()
        self.db.close()

    def test_write_and_read_position(self):
        ok = self.db.write_position(
            beacon_id="test_beacon",
            x=10.5, y=14.2, floor=2,
            confidence=0.85,
            location_label="living_room",
            raw_rssi={"living_center": -62, "kitchen_ne": -75},
            smoothed_rssi={"living_center": -63, "kitchen_ne": -74},
            n_eff=320.0,
            particle_count=500,
            floor_belief={"1": 0.01, "2": 0.98, "3": 0.01},
        )
        assert ok

        rows = self.db.read_position_history(beacon_id="test_beacon", limit=1)
        assert len(rows) == 1
        r = rows[0]
        assert r["estimated_x"] == pytest.approx(10.5)
        assert r["estimated_y"] == pytest.approx(14.2)
        assert r["estimated_floor"] == 2
        assert r["location_label"] == "living_room"
        assert r["raw_rssi"]["living_center"] == -62
        assert r["floor_belief"]["2"] == pytest.approx(0.98)

    def test_write_and_read_fingerprint(self):
        ok = self.db.write_fingerprint(
            location_label="kitchen",
            floor=2,
            grid_x=12.0, grid_y=25.0,
            rssi_vector={"kitchen_ne": -55, "living_center": -72},
            rssi_std={"kitchen_ne": 1.2, "living_center": 2.1},
            duration_seconds=60.0,
            n_readings=120,
            notes="test_fingerprint",
        )
        assert ok

        rows = self.db.read_fingerprint_samples(
            floor=2, location_label="kitchen", limit=10
        )
        assert len(rows) >= 1
        r = rows[0]
        assert r["grid_x"] == pytest.approx(12.0)
        assert r["rssi_vector"]["kitchen_ne"] == -55

    def test_count_positions(self):
        before = self.db.count_positions(beacon_id="test_count")
        self.db.write_position(
            beacon_id="test_count",
            x=1.0, y=1.0, floor=1,
            confidence=0.5,
            location_label="office",
        )
        after = self.db.count_positions(beacon_id="test_count")
        assert after == before + 1
        # Clean up this specific beacon too
        from sqlalchemy import text
        with self.db._engine.connect() as conn:
            conn.execute(
                text("DELETE FROM position_history WHERE beacon_id = 'test_count'")
            )
            conn.commit()

    def test_position_history_filter_by_floor(self):
        self.db.write_position(
            beacon_id="test_floor_filter",
            x=5.0, y=10.0, floor=1,
            confidence=0.5,
            location_label="office",
        )
        self.db.write_position(
            beacon_id="test_floor_filter",
            x=10.0, y=14.0, floor=2,
            confidence=0.7,
            location_label="living_room",
        )
        rows_f1 = self.db.read_position_history(
            beacon_id="test_floor_filter", floor=1
        )
        rows_f2 = self.db.read_position_history(
            beacon_id="test_floor_filter", floor=2
        )
        assert all(r["estimated_floor"] == 1 for r in rows_f1)
        assert all(r["estimated_floor"] == 2 for r in rows_f2)
        # Cleanup
        from sqlalchemy import text
        with self.db._engine.connect() as conn:
            conn.execute(
                text(
                    "DELETE FROM position_history "
                    "WHERE beacon_id = 'test_floor_filter'"
                )
            )
            conn.commit()

    def test_connected_flag(self):
        assert self.db.connected is True
        self.db.close()
        assert self.db.connected is False
