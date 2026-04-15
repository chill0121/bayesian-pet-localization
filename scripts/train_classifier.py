#!/usr/bin/env python3
"""
Train the Random Forest zone classifier for Bayesian Pet Localization.

Supports two data sources:
  1. CSV files (site survey exports) — no Postgres needed
  2. PostgreSQL fingerprint_samples table — uses the same DB as inference

Usage:
    # Train from survey CSVs (default: scratch/survey_f*.csv)
    python scripts/train_classifier.py --source csv

    # Train from specific CSV files
    python scripts/train_classifier.py --source csv --csv-files scratch/survey_f1.csv scratch/survey_f2.csv

    # Train from PostgreSQL
    python scripts/train_classifier.py --source db

    # Import CSVs into PostgreSQL (does NOT train)
    python scripts/train_classifier.py --import-csv

    # Import CSVs AND train from DB
    python scripts/train_classifier.py --import-csv --source db

    # Tune augmentation / estimators
    python scripts/train_classifier.py --source csv --augment-factor 8 --n-estimators 300

    # Dry-run: show label distribution without training
    python scripts/train_classifier.py --source csv --dry-run
"""

import argparse
import csv
import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow importing from services/inference/
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "services", "inference")
)

from models.classifier import ZoneClassifier

logger = logging.getLogger(__name__)

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_GLOB = str(PROJECT_ROOT / "scratch" / "survey_f*.csv")
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_LAYOUT_PATH = PROJECT_ROOT / "config" / "floorplan" / "layout.json"


# ---------------------------------------------------------------------------
# POI-based zone relabeling
# ---------------------------------------------------------------------------

def load_poi_definitions(layout_path: str | Path) -> tuple[dict, float]:
    """Load POI definitions from layout.json.

    Returns
    -------
    poi_defs : dict
        ``{(floor, room_name): [{"name": str, "x": float, "y": float}]}``
    radius_ft : float
        Default POI zone radius from layout.json (``poi_radius_ft``).
    """
    with open(layout_path) as f:
        data = json.load(f)

    radius_ft = data.get("poi_radius_ft", 3.0)
    poi_defs: dict[tuple[int, str], list[dict]] = {}

    for floor_data in data.get("floors", []):
        floor_num = floor_data["floor"]
        for room in floor_data.get("rooms", []):
            room_name = room["name"]
            pois = room.get("poi", [])
            if not pois:
                continue
            entries = []
            for p in pois:
                pos = p.get("position", [p.get("x", 0), p.get("y", 0)])
                entries.append({
                    "name": p["name"],
                    "x": pos[0],
                    "y": pos[1],
                })
            poi_defs[(floor_num, room_name)] = entries

    return poi_defs, radius_ft


def relabel_samples_with_pois(
    samples: list[dict],
    poi_defs: dict[tuple[int, str], list[dict]],
    radius_ft: float,
) -> int:
    """Relabel survey samples near POIs with ``{room}_{poi_name}`` zone labels.

    Modifies samples in-place.  The ``room`` field is preserved so that
    ``zone_to_room`` maps ``office_dog_bed`` → ``office`` automatically.

    Returns the number of samples relabeled.
    """
    import math

    relabeled = 0
    for s in samples:
        floor = s["floor"]
        room = s["room"]
        key = (floor, room)
        pois = poi_defs.get(key)
        if not pois:
            continue

        sx, sy = s["grid_x"], s["grid_y"]
        best_dist = float("inf")
        best_poi = None

        for poi in pois:
            dx = sx - poi["x"]
            dy = sy - poi["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= radius_ft and dist < best_dist:
                best_dist = dist
                best_poi = poi

        if best_poi is not None:
            new_label = f"{room}_{best_poi['name']}"
            if s["zone_label"] != new_label:
                s["zone_label"] = new_label
                relabeled += 1

    return relabeled


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_samples_from_csv(csv_paths: list[str]) -> list[dict]:
    """Load fingerprint samples from site survey CSV files.

    Converts CSV rows into the sample dict format expected by
    ``ZoneClassifier.train()``:
        - zone_label: sub-zone name (e.g. "kitchen", "living_room")
        - room: parent room name (e.g. "living_kitchen")
        - floor: int
        - rssi_vector: {anchor_id: mean_rssi}
        - rssi_std: {anchor_id: std_dev}
        - grid_x, grid_y, point_type, n_readings, duration_seconds
    """
    samples = []

    for path in csv_paths:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            # Discover anchor columns: rssi_<AnchorId>
            rssi_cols = [h for h in headers if h.startswith("rssi_")]
            std_cols = [h for h in headers if h.startswith("std_")]

            for row in reader:
                rssi_vector = {}
                rssi_std = {}

                for col in rssi_cols:
                    anchor_id = col[len("rssi_"):]  # strip "rssi_" prefix
                    val = row.get(col, "")
                    if val:
                        rssi_vector[anchor_id] = float(val)

                for col in std_cols:
                    anchor_id = col[len("std_"):]
                    val = row.get(col, "")
                    if val:
                        rssi_std[anchor_id] = float(val)

                sample = {
                    "zone_label": row.get("zone", row.get("location_label", "unknown")),
                    "room": row.get("room", row.get("zone", "unknown")),
                    "floor": int(row["floor"]),
                    "grid_x": float(row["grid_x"]),
                    "grid_y": float(row["grid_y"]),
                    "rssi_vector": rssi_vector,
                    "rssi_std": rssi_std,
                    "point_type": row.get("point_type", "grid"),
                    "n_readings": int(row.get("n_readings", 0)),
                    "duration_seconds": float(row.get("duration_seconds", 0)),
                }
                samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# PostgreSQL loading
# ---------------------------------------------------------------------------

def load_samples_from_db(
    host: str = "localhost",
    port: int = 5432,
    user: str = "localization",
    password: str = "",
    dbname: str = "pet_tracking",
) -> list[dict]:
    """Load fingerprint samples from PostgreSQL."""
    from db import Database

    db = Database(host=host, port=port, user=user, password=password, dbname=dbname)
    if not db.connect():
        logger.error("Could not connect to PostgreSQL at %s:%d", host, port)
        sys.exit(1)

    rows = db.read_fingerprint_samples(limit=10000)
    db.close()

    if not rows:
        logger.error("No fingerprint samples found in database")
        sys.exit(1)

    # DB rows already have the right keys (zone_label, room, floor, rssi_vector)
    return rows


# ---------------------------------------------------------------------------
# CSV → PostgreSQL import
# ---------------------------------------------------------------------------

def import_csv_to_db(
    csv_paths: list[str],
    host: str = "localhost",
    port: int = 5432,
    user: str = "localization",
    password: str = "",
    dbname: str = "pet_tracking",
) -> int:
    """Import survey CSV files into the fingerprint_samples table.

    Returns the number of rows successfully inserted.
    """
    from db import Database

    db = Database(host=host, port=port, user=user, password=password, dbname=dbname)
    if not db.connect():
        logger.error("Could not connect to PostgreSQL at %s:%d", host, port)
        sys.exit(1)

    samples = load_samples_from_csv(csv_paths)
    inserted = 0

    for s in samples:
        ok = db.write_fingerprint(
            location_label=s["zone_label"],
            zone_label=s["zone_label"],
            room=s["room"],
            floor=s["floor"],
            grid_x=s["grid_x"],
            grid_y=s["grid_y"],
            rssi_vector=s["rssi_vector"],
            rssi_std=s.get("rssi_std"),
            duration_seconds=s.get("duration_seconds"),
            n_readings=s.get("n_readings"),
            notes=f"imported from CSV; point_type={s.get('point_type', 'grid')}",
        )
        if ok:
            inserted += 1

    db.close()
    logger.info("Imported %d / %d samples to PostgreSQL", inserted, len(samples))
    return inserted


# ---------------------------------------------------------------------------
# Label distribution summary
# ---------------------------------------------------------------------------

def print_label_summary(samples: list[dict]) -> None:
    """Print zone/room distribution table."""
    from collections import Counter

    zone_counts = Counter()
    room_counts = Counter()
    floor_counts = Counter()

    for s in samples:
        zone = s.get("zone_label", s.get("location_label", "unknown"))
        room = s.get("room", zone)
        floor = s.get("floor", "?")
        zone_counts[zone] += 1
        room_counts[room] += 1
        floor_counts[floor] += 1

    print(f"\n{'='*50}")
    print(f"  Total samples: {len(samples)}")
    print(f"{'='*50}")

    print(f"\n  {'Floor':<10} {'Count':>6}")
    print(f"  {'-'*16}")
    for f in sorted(floor_counts):
        print(f"  {f:<10} {floor_counts[f]:>6}")

    print(f"\n  {'Room':<20} {'Count':>6}")
    print(f"  {'-'*26}")
    for r in sorted(room_counts):
        print(f"  {r:<20} {room_counts[r]:>6}")

    print(f"\n  {'Zone (label)':<20} {'Count':>6}")
    print(f"  {'-'*26}")
    for z in sorted(zone_counts):
        print(f"  {z:<20} {zone_counts[z]:>6}")
    print()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save(
    samples: list[dict],
    anchor_ids: list[str],
    model_dir: Path,
    augment_factor: int = 5,
    n_estimators: int = 200,
    cv_folds: int = 5,
    rssi_noise_std: float = 1.5,
) -> tuple[dict, Path]:
    """Train classifier and save to models/ directory.

    Returns (metrics_dict, saved_model_path).
    """
    clf = ZoneClassifier(anchor_ids=anchor_ids)

    metrics = clf.train(
        samples,
        augment_factor=augment_factor,
        n_estimators=n_estimators,
        cv_folds=cv_folds,
        rssi_noise_std=rssi_noise_std,
    )

    # Version string: YYYYMMDD_HHMMSS
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"random_forest_v{version}.joblib"
    clf.save(str(model_path))

    return metrics, model_path


def print_metrics(metrics: dict, model_path: Path) -> None:
    """Pretty-print training results."""
    print(f"\n{'='*50}")
    print(f"  Training Results")
    print(f"{'='*50}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    if metrics.get("cv_scores"):
        scores = metrics["cv_scores"]
        print(f"  CV Scores:    {scores}")
        print(f"  CV Mean±Std:  {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"  Classes:      {metrics['n_classes']}")
    print(f"  Features:     {metrics['n_features']}")

    print(f"\n  Per-class F1:")
    for cls, f1 in sorted(metrics.get("per_class_f1", {}).items()):
        print(f"    {cls:<20} {f1:.4f}")

    print(f"\n  Top features:")
    for feat, imp in list(metrics.get("top_features", {}).items())[:10]:
        print(f"    {feat:<30} {imp:.5f}")

    print(f"\n  Model saved to: {model_path}")

    # Confusion matrix
    cm = metrics.get("confusion_matrix")
    labels = metrics.get("confusion_labels", [])
    if cm and labels:
        print(f"\n  Confusion Matrix:")
        header = "  " + " " * 20 + "".join(f"{l[:8]:>9}" for l in labels)
        print(header)
        for i, row_label in enumerate(labels):
            row_str = "".join(f"{cm[i][j]:>9}" for j in range(len(labels)))
            print(f"  {row_label:<20}{row_str}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the Random Forest zone classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["csv", "db"],
        default="csv",
        help="Data source: 'csv' for survey CSV files, 'db' for PostgreSQL (default: csv)",
    )
    parser.add_argument(
        "--csv-files",
        nargs="+",
        help="Paths to survey CSV files (default: scratch/survey_f*.csv)",
    )
    parser.add_argument(
        "--import-csv",
        action="store_true",
        help="Import CSV files into PostgreSQL fingerprint_samples table",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to save trained model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=DEFAULT_LAYOUT_PATH,
        help=f"Path to layout.json for POI zone definitions (default: {DEFAULT_LAYOUT_PATH})",
    )
    parser.add_argument(
        "--no-poi-relabel",
        action="store_true",
        help="Skip POI-based zone relabeling (use raw survey zone labels)",
    )
    parser.add_argument(
        "--augment-factor",
        type=int,
        default=5,
        help="Data augmentation multiplier (default: 5)",
    )
    parser.add_argument(
        "--rssi-noise-std",
        type=float,
        default=1.5,
        help="Std-dev of Gaussian noise for augmentation (default: 1.5)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the Random Forest (default: 200)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show label distribution without training",
    )

    # DB connection args
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-user", default="localization")
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-name", default="pet_tracking")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- Resolve CSV paths ---
    csv_paths = args.csv_files
    if csv_paths is None:
        csv_paths = sorted(glob.glob(DEFAULT_CSV_GLOB))
        if not csv_paths and args.source == "csv":
            logger.error("No CSV files found matching %s", DEFAULT_CSV_GLOB)
            sys.exit(1)

    # --- Import CSVs to DB (if requested) ---
    if args.import_csv:
        if not csv_paths:
            logger.error("No CSV files to import")
            sys.exit(1)
        print(f"Importing {len(csv_paths)} CSV files to PostgreSQL...")
        for p in csv_paths:
            print(f"  {p}")
        import_csv_to_db(
            csv_paths,
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_password,
            dbname=args.db_name,
        )
        if args.source != "db":
            print("Import complete. Use --source db to train from DB data.")
            return

    # --- Load samples ---
    if args.source == "csv":
        if not csv_paths:
            logger.error("No CSV files found")
            sys.exit(1)
        print(f"Loading samples from {len(csv_paths)} CSV files...")
        for p in csv_paths:
            print(f"  {p}")
        samples = load_samples_from_csv(csv_paths)
    else:
        print("Loading samples from PostgreSQL...")
        samples = load_samples_from_db(
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_password,
            dbname=args.db_name,
        )

    if not samples:
        logger.error("No samples loaded")
        sys.exit(1)

    # --- POI-based zone relabeling ---
    if not args.no_poi_relabel and args.layout.exists():
        poi_defs, poi_radius = load_poi_definitions(args.layout)
        if poi_defs:
            n_relabeled = relabel_samples_with_pois(samples, poi_defs, poi_radius)
            n_pois = sum(len(v) for v in poi_defs.values())
            print(f"\nPOI relabeling: {n_relabeled} samples relabeled "
                  f"({n_pois} POIs, radius={poi_radius} ft)")
        else:
            print("\nNo POI definitions found in layout.json — skipping relabeling")
    elif not args.no_poi_relabel:
        print(f"\nLayout file not found at {args.layout} — skipping POI relabeling")

    print_label_summary(samples)

    if args.dry_run:
        print("Dry run — skipping training.")
        return

    # --- Discover anchor IDs from the data ---
    anchor_ids = set()
    for s in samples:
        anchor_ids.update(s.get("rssi_vector", {}).keys())
    anchor_ids = sorted(anchor_ids)
    print(f"Anchors ({len(anchor_ids)}): {anchor_ids}")

    # --- Train ---
    print(f"\nTraining with augment_factor={args.augment_factor}, "
          f"rssi_noise_std={args.rssi_noise_std}, "
          f"n_estimators={args.n_estimators}, cv_folds={args.cv_folds}...")

    metrics, model_path = train_and_save(
        samples=samples,
        anchor_ids=anchor_ids,
        model_dir=args.model_dir,
        augment_factor=args.augment_factor,
        n_estimators=args.n_estimators,
        cv_folds=args.cv_folds,
        rssi_noise_std=args.rssi_noise_std,
    )

    print_metrics(metrics, model_path)


if __name__ == "__main__":
    main()
