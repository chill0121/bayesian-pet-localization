"""
PostgreSQL integration for the inference service.

Uses SQLAlchemy Core (not ORM) for lightweight, thread-safe database access.
Follows the same resilient pattern as the InfluxDB client: silent degradation
when Postgres is unavailable, with periodic reconnect attempts.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    desc,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table definitions (mirror init.sql)
# ---------------------------------------------------------------------------

metadata = MetaData()

position_history = Table(
    "position_history",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("timestamp", TIMESTAMP(timezone=True)),
    Column("beacon_id", String(100), nullable=False),
    Column("estimated_x", Float, nullable=False),
    Column("estimated_y", Float, nullable=False),
    Column("estimated_floor", Integer, nullable=False),
    Column("confidence", Float),
    Column("location_label", String(50)),
    Column("activity_state", String(20)),
    Column("raw_rssi", JSONB),
    Column("smoothed_rssi", JSONB),
    Column("n_eff", Float),
    Column("particle_count", Integer),
    Column("floor_belief", JSONB),
)

fingerprint_samples = Table(
    "fingerprint_samples",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", TIMESTAMP(timezone=True)),
    Column("location_label", String(50), nullable=False),
    Column("floor", Integer, nullable=False),
    Column("grid_x", Float, nullable=False),
    Column("grid_y", Float, nullable=False),
    Column("rssi_vector", JSONB, nullable=False),
    Column("rssi_std", JSONB),
    Column("features", JSONB),
    Column("duration_seconds", Float),
    Column("n_readings", Integer),
    Column("notes", Text),
)

model_versions = Table(
    "model_versions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model_type", String(50), nullable=False),
    Column("version", String(20), nullable=False),
    Column("artifact_path", String(200)),
    Column("metrics", JSONB),
    Column("hyperparameters", JSONB),
    Column("n_training_samples", Integer),
    Column("trained_at", TIMESTAMP(timezone=True)),
    Column("active", Boolean, default=False),
)

# ---------------------------------------------------------------------------
# Database configuration constants
# ---------------------------------------------------------------------------

_RETRY_INTERVAL: float = 60.0    # seconds between reconnect attempts
DB_POOL_SIZE: int = 2            # base connection pool size
DB_POOL_MAX_OVERFLOW: int = 3    # additional connections beyond pool_size
DB_POOL_RECYCLE_SECONDS: int = 300  # recycle connections after this many seconds


class Database:
    """Thread-safe Postgres client with resilient connection handling."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "localization",
        password: str = "",
        dbname: str = "pet_tracking",
    ):
        self._dsn = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        self._engine: Optional[Engine] = None
        self._last_attempt: float = 0.0

    # -- connection management -----------------------------------------------

    @property
    def connected(self) -> bool:
        return self._engine is not None

    def connect(self) -> bool:
        """Try to connect; returns True on success.  Respects retry backoff."""
        now = time.time()
        if now - self._last_attempt < _RETRY_INTERVAL:
            return self.connected
        self._last_attempt = now
        try:
            self._engine = create_engine(
                self._dsn,
                poolclass=QueuePool,
                pool_size=DB_POOL_SIZE,
                max_overflow=DB_POOL_MAX_OVERFLOW,
                pool_pre_ping=True,  # verify connections before use
                pool_recycle=DB_POOL_RECYCLE_SECONDS,
            )
            # Verify by opening a real connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected to PostgreSQL")
            return True
        except Exception as e:
            logger.info(
                "PostgreSQL not available — position history will not be "
                "persisted (retry in %ds): %s",
                int(_RETRY_INTERVAL),
                e,
            )
            self._engine = None
            return False

    def close(self):
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    # -- writes ---------------------------------------------------------------

    def write_position(
        self,
        beacon_id: str,
        x: float,
        y: float,
        floor: int,
        confidence: float,
        location_label: str,
        activity_state: str = "unknown",
        raw_rssi: Optional[dict] = None,
        smoothed_rssi: Optional[dict] = None,
        n_eff: Optional[float] = None,
        particle_count: Optional[int] = None,
        floor_belief: Optional[dict] = None,
    ) -> bool:
        """Insert a position record.  Returns True on success."""
        if self._engine is None:
            if not self.connect():
                return False
        try:
            with self._engine.connect() as conn:
                conn.execute(
                    position_history.insert().values(
                        timestamp=datetime.now(timezone.utc),
                        beacon_id=beacon_id,
                        estimated_x=x,
                        estimated_y=y,
                        estimated_floor=floor,
                        confidence=confidence,
                        location_label=location_label,
                        activity_state=activity_state,
                        raw_rssi=raw_rssi,
                        smoothed_rssi=smoothed_rssi,
                        n_eff=n_eff,
                        particle_count=particle_count,
                        floor_belief=floor_belief,
                    )
                )
                conn.commit()
            return True
        except Exception as e:
            logger.warning("PostgreSQL write failed: %s", e)
            self._engine = None  # force reconnect on next attempt
            return False

    def write_fingerprint(
        self,
        location_label: str,
        floor: int,
        grid_x: float,
        grid_y: float,
        rssi_vector: dict,
        rssi_std: Optional[dict] = None,
        features: Optional[dict] = None,
        duration_seconds: Optional[float] = None,
        n_readings: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Insert a site-survey fingerprint sample.  Returns True on success."""
        if self._engine is None:
            if not self.connect():
                return False
        try:
            with self._engine.connect() as conn:
                conn.execute(
                    fingerprint_samples.insert().values(
                        timestamp=datetime.now(timezone.utc),
                        location_label=location_label,
                        floor=floor,
                        grid_x=grid_x,
                        grid_y=grid_y,
                        rssi_vector=rssi_vector,
                        rssi_std=rssi_std,
                        features=features,
                        duration_seconds=duration_seconds,
                        n_readings=n_readings,
                        notes=notes,
                    )
                )
                conn.commit()
            return True
        except Exception as e:
            logger.warning("PostgreSQL fingerprint write failed: %s", e)
            self._engine = None
            return False

    # -- reads ----------------------------------------------------------------

    def read_position_history(
        self,
        beacon_id: Optional[str] = None,
        floor: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return recent position history, newest first."""
        if self._engine is None:
            if not self.connect():
                return []
        try:
            query = position_history.select().order_by(
                desc(position_history.c.timestamp)
            )
            if beacon_id is not None:
                query = query.where(position_history.c.beacon_id == beacon_id)
            if floor is not None:
                query = query.where(position_history.c.estimated_floor == floor)
            query = query.limit(limit)

            with self._engine.connect() as conn:
                rows = conn.execute(query).mappings().all()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("PostgreSQL read failed: %s", e)
            self._engine = None
            return []

    def read_fingerprint_samples(
        self,
        floor: Optional[int] = None,
        location_label: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Return fingerprint samples for model training."""
        if self._engine is None:
            if not self.connect():
                return []
        try:
            query = fingerprint_samples.select().order_by(
                desc(fingerprint_samples.c.timestamp)
            )
            if floor is not None:
                query = query.where(fingerprint_samples.c.floor == floor)
            if location_label is not None:
                query = query.where(
                    fingerprint_samples.c.location_label == location_label
                )
            query = query.limit(limit)

            with self._engine.connect() as conn:
                rows = conn.execute(query).mappings().all()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("PostgreSQL fingerprint read failed: %s", e)
            self._engine = None
            return []

    def count_positions(self, beacon_id: Optional[str] = None) -> int:
        """Return total position_history row count."""
        if self._engine is None:
            if not self.connect():
                return 0
        try:
            q = text("SELECT COUNT(*) FROM position_history")
            params = {}
            if beacon_id is not None:
                q = text(
                    "SELECT COUNT(*) FROM position_history WHERE beacon_id = :bid"
                )
                params = {"bid": beacon_id}
            with self._engine.connect() as conn:
                return conn.execute(q, params).scalar() or 0
        except Exception as e:
            logger.warning("PostgreSQL count failed: %s", e)
            self._engine = None
            return 0

    def count_fingerprints(self) -> int:
        """Return total fingerprint_samples row count."""
        if self._engine is None:
            if not self.connect():
                return 0
        try:
            with self._engine.connect() as conn:
                return (
                    conn.execute(
                        text("SELECT COUNT(*) FROM fingerprint_samples")
                    ).scalar()
                    or 0
                )
        except Exception as e:
            logger.warning("PostgreSQL count failed: %s", e)
            self._engine = None
            return 0
