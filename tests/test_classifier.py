"""
Tests for the Random Forest zone classifier.

Run with: python -m pytest tests/test_classifier.py -v
"""

import sys
import os
import tempfile

import numpy as np
import pytest

# Allow importing from services/inference/
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "services", "inference")
)

from models.classifier import ZoneClassifier, RoomClassifier, MISSING_RSSI_SENTINEL

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

ANCHOR_IDS = [
    "1F_Hallway",
    "1F_Office",
    "kitchen_ne",
    "living_center",
    "living_sw",
    "staircase_mid",
    "3F_hallway",
    "3F_master_bed",
]

# Synthetic fingerprint samples that mimic DB rows from fingerprint_samples
# Each sample: zone_label, room, floor, rssi_vector dict
# Office has two sub-zones (office_desk, office_dog_bed) to test scoping.
SYNTHETIC_SAMPLES = [
    # office — sub-zone: office_desk (closer to 1F_Office anchor)
    {"zone_label": "office_desk", "room": "office", "floor": 1,
     "rssi_vector": {"1F_Office": -45, "1F_Hallway": -62, "living_sw": -80, "staircase_mid": -82},
     "features": {}},
    {"zone_label": "office_desk", "room": "office", "floor": 1,
     "rssi_vector": {"1F_Office": -47, "1F_Hallway": -64, "living_sw": -81, "staircase_mid": -84},
     "features": {}},
    {"zone_label": "office_desk", "room": "office", "floor": 1,
     "rssi_vector": {"1F_Office": -44, "1F_Hallway": -60, "living_sw": -79, "staircase_mid": -83},
     "features": {}},
    {"zone_label": "office_desk", "room": "office", "floor": 1,
     "rssi_vector": {"1F_Office": -46, "1F_Hallway": -63, "living_sw": -82, "staircase_mid": -85},
     "features": {}},
    # office — sub-zone: office_dog_bed (between office and hallway anchors)
    {"zone_label": "office_dog_bed", "room": "office", "floor": 1,
     "rssi_vector": {"1F_Office": -53, "1F_Hallway": -54, "living_sw": -79, "staircase_mid": -78},
     "features": {}},
    {"zone_label": "office_dog_bed", "room": "office", "floor": 1,
     "rssi_vector": {"1F_Office": -55, "1F_Hallway": -56, "living_sw": -80, "staircase_mid": -80},
     "features": {}},
    {"zone_label": "office_dog_bed", "room": "office", "floor": 1,
     "rssi_vector": {"1F_Office": -52, "1F_Hallway": -53, "living_sw": -78, "staircase_mid": -77},
     "features": {}},
    {"zone_label": "office_dog_bed", "room": "office", "floor": 1,
     "rssi_vector": {"1F_Office": -54, "1F_Hallway": -55, "living_sw": -81, "staircase_mid": -79},
     "features": {}},
    # hallway — single zone (zone == room)
    {"zone_label": "hallway", "room": "hallway", "floor": 1,
     "rssi_vector": {"1F_Office": -62, "1F_Hallway": -44, "living_sw": -78, "staircase_mid": -70},
     "features": {}},
    {"zone_label": "hallway", "room": "hallway", "floor": 1,
     "rssi_vector": {"1F_Office": -64, "1F_Hallway": -46, "living_sw": -79, "staircase_mid": -72},
     "features": {}},
    {"zone_label": "hallway", "room": "hallway", "floor": 1,
     "rssi_vector": {"1F_Office": -60, "1F_Hallway": -43, "living_sw": -77, "staircase_mid": -71},
     "features": {}},
    {"zone_label": "hallway", "room": "hallway", "floor": 1,
     "rssi_vector": {"1F_Office": -63, "1F_Hallway": -45, "living_sw": -80, "staircase_mid": -73},
     "features": {}},
    # kitchen — single zone
    {"zone_label": "kitchen", "room": "kitchen", "floor": 2,
     "rssi_vector": {"kitchen_ne": -42, "living_center": -58, "living_sw": -65, "staircase_mid": -60, "1F_Office": -85},
     "features": {}},
    {"zone_label": "kitchen", "room": "kitchen", "floor": 2,
     "rssi_vector": {"kitchen_ne": -44, "living_center": -60, "living_sw": -67, "staircase_mid": -62, "1F_Office": -87},
     "features": {}},
    {"zone_label": "kitchen", "room": "kitchen", "floor": 2,
     "rssi_vector": {"kitchen_ne": -41, "living_center": -57, "living_sw": -64, "staircase_mid": -59, "1F_Office": -84},
     "features": {}},
    {"zone_label": "kitchen", "room": "kitchen", "floor": 2,
     "rssi_vector": {"kitchen_ne": -43, "living_center": -59, "living_sw": -66, "staircase_mid": -61, "1F_Office": -86},
     "features": {}},
    # living_room — single zone
    {"zone_label": "living_room", "room": "living_room", "floor": 2,
     "rssi_vector": {"living_sw": -40, "living_center": -50, "kitchen_ne": -62, "staircase_mid": -58, "1F_Office": -83},
     "features": {}},
    {"zone_label": "living_room", "room": "living_room", "floor": 2,
     "rssi_vector": {"living_sw": -42, "living_center": -52, "kitchen_ne": -64, "staircase_mid": -60, "1F_Office": -85},
     "features": {}},
    {"zone_label": "living_room", "room": "living_room", "floor": 2,
     "rssi_vector": {"living_sw": -39, "living_center": -49, "kitchen_ne": -61, "staircase_mid": -57, "1F_Office": -82},
     "features": {}},
    {"zone_label": "living_room", "room": "living_room", "floor": 2,
     "rssi_vector": {"living_sw": -41, "living_center": -51, "kitchen_ne": -63, "staircase_mid": -59, "1F_Office": -84},
     "features": {}},
    # master_bed — single zone
    {"zone_label": "master_bed", "room": "master_bed", "floor": 3,
     "rssi_vector": {"3F_master_bed": -43, "3F_hallway": -58, "staircase_mid": -80, "1F_Office": -90},
     "features": {}},
    {"zone_label": "master_bed", "room": "master_bed", "floor": 3,
     "rssi_vector": {"3F_master_bed": -45, "3F_hallway": -60, "staircase_mid": -82, "1F_Office": -92},
     "features": {}},
    {"zone_label": "master_bed", "room": "master_bed", "floor": 3,
     "rssi_vector": {"3F_master_bed": -44, "3F_hallway": -59, "staircase_mid": -81, "1F_Office": -91},
     "features": {}},
    {"zone_label": "master_bed", "room": "master_bed", "floor": 3,
     "rssi_vector": {"3F_master_bed": -46, "3F_hallway": -61, "staircase_mid": -83, "1F_Office": -93},
     "features": {}},
]


@pytest.fixture
def classifier():
    """Untrained classifier with standard anchor IDs."""
    return ZoneClassifier(anchor_ids=ANCHOR_IDS)


@pytest.fixture
def trained_classifier():
    """Classifier trained on synthetic data."""
    clf = ZoneClassifier(anchor_ids=ANCHOR_IDS)
    clf.train(SYNTHETIC_SAMPLES, augment_factor=3, cv_folds=3)
    return clf


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_anchor_ids_sorted(self, classifier):
        """Anchor IDs are stored in sorted order for reproducibility."""
        assert classifier.anchor_ids == sorted(ANCHOR_IDS)

    def test_feature_names_count(self, classifier):
        """Feature vector has expected length: 8 anchors × 4 groups + aggregates."""
        names = classifier.feature_names
        n_anchors = len(ANCHOR_IDS)
        # rssi(8) + delta(8+2) + rank(8+2) + var(8+2) + floor(6+2) + composite(2)
        expected = n_anchors + (n_anchors + 2) + (n_anchors + 2) + (n_anchors + 2) + 8 + 2
        assert len(names) == expected

    def test_repr_untrained(self, classifier):
        r = repr(classifier)
        assert "untrained" in r
        assert f"anchors={len(ANCHOR_IDS)}" in r
        assert "zones=0" in r

    def test_not_trained(self, classifier):
        assert not classifier.is_trained

    def test_classes_empty_before_training(self, classifier):
        assert classifier.classes == []

    def test_feature_importances_empty_before_training(self, classifier):
        assert classifier.feature_importances == {}


# ---------------------------------------------------------------------------
# Feature vector assembly
# ---------------------------------------------------------------------------


class TestFeatureVector:
    def test_vector_length(self, classifier):
        """Assembled vector matches feature_names length."""
        features = {"delta_mean": 1.5, "delta_max": 3.0}
        rssi = {"1F_Office": -55, "living_sw": -70}
        vec = classifier._assemble_vector(features, rssi)
        assert len(vec) == len(classifier.feature_names)

    def test_missing_anchor_gets_sentinel(self, classifier):
        """Anchors not in smoothed_rssi get the sentinel value."""
        rssi = {"1F_Office": -55}  # only one anchor
        vec = classifier._assemble_vector({}, rssi)
        names = classifier.feature_names
        # All other rssi_<anchor> should be sentinel
        for i, name in enumerate(names):
            if name.startswith("rssi_") and not name.startswith(("rssi_mean_floor_", "rssi_best_floor", "rssi_gap_")):
                aid = name[len("rssi_"):]
                if aid == "1F_Office":
                    assert vec[i] == -55.0
                else:
                    assert vec[i] == MISSING_RSSI_SENTINEL

    def test_features_mapped_correctly(self, classifier):
        """Feature dict values land in the correct positions."""
        features = {
            "delta_mean": 2.5,
            "delta_max": 4.0,
            "var_mean": 10.0,
            "activity_score": 0.7,
            "n_reporting": 5,
        }
        vec = classifier._assemble_vector(features)
        names = classifier.feature_names
        assert vec[names.index("delta_mean")] == 2.5
        assert vec[names.index("delta_max")] == 4.0
        assert vec[names.index("var_mean")] == 10.0
        assert vec[names.index("activity_score")] == 0.7
        assert vec[names.index("n_reporting")] == 5

    def test_all_zeros_for_empty_input(self, classifier):
        """Empty dicts produce sentinel for RSSI, zero for everything else."""
        vec = classifier._assemble_vector({}, {})
        names = classifier.feature_names
        for i, name in enumerate(names):
            if name.startswith("rssi_") and not name.startswith(("rssi_mean_floor_", "rssi_best_floor", "rssi_gap_")):
                assert vec[i] == MISSING_RSSI_SENTINEL
            elif name.startswith("rssi_mean_floor_"):
                assert vec[i] == MISSING_RSSI_SENTINEL


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:
    def test_augmentation_size(self):
        """Augmentation produces (1 + factor) × N samples."""
        X = np.random.randn(10, 20)
        y = np.array(["a"] * 5 + ["b"] * 5)
        X_aug, y_aug = ZoneClassifier._augment(X, y, anchor_count=8, factor=3)
        assert X_aug.shape[0] == 10 * (1 + 3)
        assert y_aug.shape[0] == 10 * (1 + 3)

    def test_original_data_preserved(self):
        """First N rows of augmented data are the originals."""
        X = np.random.randn(5, 10)
        y = np.array(["a", "b", "c", "a", "b"])
        X_aug, y_aug = ZoneClassifier._augment(X, y, anchor_count=3, factor=2)
        np.testing.assert_array_equal(X_aug[:5], X)
        np.testing.assert_array_equal(y_aug[:5], y)

    def test_sentinel_not_noised(self):
        """Sentinel values should not be altered by RSSI noise."""
        X = np.full((4, 6), MISSING_RSSI_SENTINEL)
        y = np.array(["a", "a", "b", "b"])
        X_aug, _ = ZoneClassifier._augment(
            X, y, anchor_count=6, factor=2, rssi_noise_std=5.0, dropout_prob=0.0
        )
        # All augmented RSSI columns should still be sentinel (noise × 0 mask)
        np.testing.assert_array_equal(X_aug[:, :6], MISSING_RSSI_SENTINEL)

    def test_reproducibility(self):
        """Same random state produces identical augmentation."""
        X = np.random.randn(10, 15)
        y = np.array(["a"] * 5 + ["b"] * 5)
        X1, y1 = ZoneClassifier._augment(X, y, 5, factor=3, rng=np.random.default_rng(99))
        X2, y2 = ZoneClassifier._augment(X, y, 5, factor=3, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTraining:
    def test_train_returns_metrics(self, trained_classifier):
        """Training returns a metrics dict with expected keys."""
        # Re-train to capture metrics
        metrics = trained_classifier.train(
            SYNTHETIC_SAMPLES, augment_factor=3, cv_folds=3
        )
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "per_class_f1" in metrics
        assert "confusion_matrix" in metrics
        assert "top_features" in metrics
        assert "n_classes" in metrics
        assert "n_features" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_trained_state(self, trained_classifier):
        assert trained_classifier.is_trained
        assert len(trained_classifier.classes) > 0

    def test_zone_to_room_populated(self, trained_classifier):
        """Training populates the zone_to_room mapping."""
        ztm = trained_classifier.zone_to_room
        assert len(ztm) > 0
        # Every class label should be in the mapping
        for cls in trained_classifier.classes:
            assert cls in ztm
        # Sub-zones map to their parent room
        assert ztm["office_desk"] == "office"
        assert ztm["office_dog_bed"] == "office"
        # Degenerate zones map to themselves
        assert ztm["kitchen"] == "kitchen"
        assert ztm["hallway"] == "hallway"

    def test_feature_importances_populated(self, trained_classifier):
        imps = trained_classifier.feature_importances
        assert len(imps) > 0
        # Importances should sum to ~1.0
        assert abs(sum(imps.values()) - 1.0) < 0.01

    def test_train_too_few_samples_raises(self, classifier):
        with pytest.raises(ValueError, match="at least 2"):
            classifier.train([SYNTHETIC_SAMPLES[0]])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_predict_returns_tuple(self, trained_classifier):
        """predict() returns (label, confidence, probabilities)."""
        rssi = {"1F_Office": -45, "1F_Hallway": -60, "living_sw": -80}
        label, conf, probs = trained_classifier.predict({}, smoothed_rssi=rssi)
        assert isinstance(label, str)
        assert isinstance(conf, float)
        assert isinstance(probs, dict)
        assert 0.0 <= conf <= 1.0

    def test_probabilities_sum_to_one(self, trained_classifier):
        rssi = {"kitchen_ne": -42, "living_center": -58, "living_sw": -65}
        _, _, probs = trained_classifier.predict({}, smoothed_rssi=rssi)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_office_signal(self, trained_classifier):
        """Strong 1F_Office signal should predict an office sub-zone."""
        rssi = {
            "1F_Office": -44,
            "1F_Hallway": -62,
            "living_sw": -82,
            "staircase_mid": -84,
        }
        label, conf, _ = trained_classifier.predict({}, smoothed_rssi=rssi)
        assert label in ("office_desk", "office_dog_bed")

    def test_predict_kitchen_signal(self, trained_classifier):
        """Strong kitchen_ne signal should predict kitchen."""
        rssi = {
            "kitchen_ne": -41,
            "living_center": -57,
            "living_sw": -64,
            "staircase_mid": -59,
            "1F_Office": -84,
        }
        label, conf, _ = trained_classifier.predict({}, smoothed_rssi=rssi)
        assert label == "kitchen"

    def test_predict_master_bed_signal(self, trained_classifier):
        """Strong 3F_master_bed signal should predict master_bed."""
        rssi = {
            "3F_master_bed": -43,
            "3F_hallway": -58,
            "staircase_mid": -80,
            "1F_Office": -90,
        }
        label, conf, _ = trained_classifier.predict({}, smoothed_rssi=rssi)
        assert label == "master_bed"

    def test_predict_with_no_model_raises(self, classifier):
        with pytest.raises(RuntimeError, match="No model loaded"):
            classifier.predict({}, smoothed_rssi={"1F_Office": -50})

    def test_predict_with_all_missing(self, trained_classifier):
        """Prediction still works with all-sentinel input (no anchors)."""
        label, conf, probs = trained_classifier.predict({}, smoothed_rssi={})
        assert isinstance(label, str)
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Hierarchical prediction (predict_for_room)
# ---------------------------------------------------------------------------


class TestPredictForRoom:
    def test_scopes_to_room(self, trained_classifier):
        """predict_for_room() returns only zones within the given room."""
        rssi = {"1F_Office": -45, "1F_Hallway": -60}
        label, conf, probs = trained_classifier.predict_for_room(
            {}, smoothed_rssi=rssi, room="office"
        )
        assert label in ("office_desk", "office_dog_bed")
        for z in probs:
            assert trained_classifier.zone_to_room[z] == "office"

    def test_probabilities_sum_to_one(self, trained_classifier):
        """Scoped probabilities sum to ~1.0 after renormalisation."""
        rssi = {"1F_Office": -45, "1F_Hallway": -60}
        _, _, probs = trained_classifier.predict_for_room(
            {}, smoothed_rssi=rssi, room="office"
        )
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_unknown_room_returns_none(self, trained_classifier):
        """Room with no zones in training data returns (None, 0, {})."""
        rssi = {"1F_Office": -50}
        label, conf, probs = trained_classifier.predict_for_room(
            {}, smoothed_rssi=rssi, room="basement"
        )
        assert label is None
        assert conf == 0.0
        assert probs == {}

    def test_single_zone_room_returns_zone(self, trained_classifier):
        """Room with one zone returns that zone with high confidence."""
        rssi = {"kitchen_ne": -41, "living_center": -57, "living_sw": -64}
        label, conf, probs = trained_classifier.predict_for_room(
            {}, smoothed_rssi=rssi, room="kitchen"
        )
        assert label == "kitchen"
        assert conf > 0.0
        assert "kitchen" in probs


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, trained_classifier):
        """Round-trip: save → load → predict gives same result."""
        rssi = {"1F_Office": -45, "1F_Hallway": -60}
        label1, conf1, probs1 = trained_classifier.predict({}, smoothed_rssi=rssi)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.joblib")
            trained_classifier.save(path)
            assert os.path.exists(path)

            # Load into fresh classifier
            loaded = ZoneClassifier(anchor_ids=ANCHOR_IDS)
            loaded.load(path)

            label2, conf2, probs2 = loaded.predict({}, smoothed_rssi=rssi)
            assert label1 == label2
            assert abs(conf1 - conf2) < 1e-10
            assert probs1 == probs2

    def test_from_file_factory(self, trained_classifier):
        """from_file() creates a working classifier from a saved model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            trained_classifier.save(path)

            loaded = ZoneClassifier.from_file(path)
            assert loaded.is_trained
            assert loaded.classes == trained_classifier.classes
            assert loaded.anchor_ids == trained_classifier.anchor_ids

    def test_zone_to_room_survives_roundtrip(self, trained_classifier):
        """zone_to_room mapping persists through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            trained_classifier.save(path)

            loaded = ZoneClassifier.from_file(path)
            assert loaded.zone_to_room == trained_classifier.zone_to_room

    def test_save_before_training_raises(self, classifier):
        with pytest.raises(RuntimeError, match="No model to save"):
            classifier.save("/tmp/should_not_exist.joblib")

    def test_load_restores_metadata(self, trained_classifier):
        """Loaded model preserves anchor_ids, feature_names, classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            trained_classifier.save(path)

            loaded = ZoneClassifier.from_file(path)
            assert loaded.feature_names == trained_classifier.feature_names
            assert loaded.anchor_ids == trained_classifier.anchor_ids
            assert loaded.classes == trained_classifier.classes


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_partial_anchor_coverage(self, trained_classifier):
        """Prediction works with only a subset of anchors reporting."""
        rssi = {"living_sw": -40}  # only one anchor
        label, conf, probs = trained_classifier.predict({}, smoothed_rssi=rssi)
        assert isinstance(label, str)

    def test_unknown_anchor_in_rssi(self, trained_classifier):
        """Unknown anchors in smoothed_rssi are silently ignored."""
        rssi = {"1F_Office": -45, "UNKNOWN_ANCHOR": -50}
        label, conf, probs = trained_classifier.predict({}, smoothed_rssi=rssi)
        assert isinstance(label, str)

    def test_extra_features_ignored(self, trained_classifier):
        """Extra keys in feature dict don't cause errors."""
        features = {"delta_mean": 1.0, "unknown_feature_xyz": 999}
        rssi = {"1F_Office": -50}
        label, conf, probs = trained_classifier.predict(features, smoothed_rssi=rssi)
        assert isinstance(label, str)

    def test_repr_trained(self, trained_classifier):
        r = repr(trained_classifier)
        assert "trained" in r
        assert "zones=" in r
        assert "rooms=" in r

    def test_backwards_compat_alias(self):
        """RoomClassifier is an alias for ZoneClassifier."""
        assert RoomClassifier is ZoneClassifier

    def test_location_label_fallback(self):
        """Training falls back to location_label when zone_label is absent."""
        legacy_samples = [
            {"location_label": "office", "floor": 1,
             "rssi_vector": {"1F_Office": -45, "1F_Hallway": -60},
             "features": {}},
            {"location_label": "office", "floor": 1,
             "rssi_vector": {"1F_Office": -47, "1F_Hallway": -62},
             "features": {}},
            {"location_label": "hallway", "floor": 1,
             "rssi_vector": {"1F_Office": -64, "1F_Hallway": -44},
             "features": {}},
            {"location_label": "hallway", "floor": 1,
             "rssi_vector": {"1F_Office": -62, "1F_Hallway": -43},
             "features": {}},
        ]
        clf = ZoneClassifier(anchor_ids=ANCHOR_IDS)
        clf.train(legacy_samples, augment_factor=2, cv_folds=2)
        assert "office" in clf.classes
        assert "hallway" in clf.classes
        # zone == room when no explicit room field
        assert clf.zone_to_room["office"] == "office"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
