"""
Random Forest Sub-Zone Classifier for Bayesian Pet Localization

Predicts sub-zone labels (e.g. "office_dog_bed", "kitchen_peninsula")
from RSSI-derived feature vectors produced by :class:`features.FeatureEngine`.

Designed for hierarchical fusion: the particle filter determines the room
via polygon lookup, then this classifier refines to a sub-zone within
that room.  Each zone label maps to a parent room via the ``zone_to_room``
mapping stored with the model.

Training
--------
Load fingerprint samples from PostgreSQL, extract features, optionally
augment, then fit a :class:`sklearn.ensemble.RandomForestClassifier`.
Samples provide a ``zone_label`` (sub-zone name), ``room`` (parent room),
and ``floor`` (for feature context).

Prediction
----------
Given a feature dict (from ``FeatureEngine.update()``), assemble the
feature vector in canonical order, impute missing anchors with sentinel
values.  Use ``predict_for_room()`` for hierarchical fusion (filters and
renormalizes probabilities to zones within the particle-determined room).

Persistence
-----------
Trained models are serialized via ``joblib`` to the ``models/`` directory.
A ``model_versions`` row in PostgreSQL tracks the active model.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)

# Sentinel value for missing anchor RSSI (well below real range of -30…-90)
MISSING_RSSI_SENTINEL: float = -100.0


class ZoneClassifier:
    """Random Forest sub-zone classifier.

    Predicts sub-zone labels (e.g. ``"office_dog_bed"``,
    ``"kitchen_peninsula"``) from RSSI-derived feature vectors.
    Each zone maps to a parent room via :attr:`zone_to_room`.  Use
    :meth:`predict_for_room` for hierarchical fusion where the
    particle filter determines the room.

    Parameters
    ----------
    anchor_ids : list[str]
        Ordered list of expected anchor identifiers.  Determines the
        canonical feature vector layout.
    model_path : str | None
        If given, load a pre-trained model from this path at init.
    """

    def __init__(
        self,
        anchor_ids: list[str],
        model_path: Optional[str] = None,
    ):
        self._anchor_ids: list[str] = sorted(anchor_ids)
        self._feature_names: list[str] = self._build_feature_names()
        self._model: Optional[RandomForestClassifier] = None
        self._classes: Optional[np.ndarray] = None  # zone labels
        self._zone_to_room: dict[str, str] = {}  # zone_label → parent room

        if model_path is not None:
            self.load(model_path)

    # ------------------------------------------------------------------
    # Feature vector assembly
    # ------------------------------------------------------------------

    def _build_feature_names(self) -> list[str]:
        """Return the canonical ordered list of feature names.

        This must match the order used during training *and* prediction.
        """
        names: list[str] = []

        # 1. Smoothed RSSI per anchor (passed separately, not from FeatureEngine)
        for aid in self._anchor_ids:
            names.append(f"rssi_{aid}")

        # 2. Delta RSSI per anchor + aggregates
        for aid in self._anchor_ids:
            names.append(f"delta_rssi_{aid}")
        names.extend(["delta_mean", "delta_max"])

        # 3. Anchor rankings per anchor + aggregates
        for aid in self._anchor_ids:
            names.append(f"rank_{aid}")
        names.extend(["rank_changes", "rssi_gap_1_2"])

        # 4. Rolling variance per anchor + aggregates
        for aid in self._anchor_ids:
            names.append(f"var_{aid}")
        names.extend(["var_mean", "var_max"])

        # 5. Cross-floor attenuation (floors 1-3)
        for floor in (1, 2, 3):
            names.append(f"n_anchors_floor_{floor}")
            names.append(f"rssi_mean_floor_{floor}")
        names.extend(["rssi_best_floor", "same_vs_cross_ratio"])

        # 6. Composite
        names.extend(["n_reporting", "activity_score"])

        return names

    def _assemble_vector(
        self,
        features: dict,
        smoothed_rssi: Optional[dict] = None,
    ) -> np.ndarray:
        """Convert a feature dict into an ordered numpy vector.

        Missing features are filled with appropriate sentinel/default values:
        - RSSI features → ``MISSING_RSSI_SENTINEL`` (-100)
        - Delta/variance/rank features → 0
        - Floor count features → 0
        - Floor mean RSSI features → ``MISSING_RSSI_SENTINEL``

        Parameters
        ----------
        features : dict
            Output of ``FeatureEngine.update()``.
        smoothed_rssi : dict | None
            anchor_id → smoothed RSSI.  Provides the ``rssi_<anchor>``
            features that are *not* part of the FeatureEngine output.
        """
        smoothed = smoothed_rssi or {}
        vec = np.zeros(len(self._feature_names), dtype=np.float64)

        for i, name in enumerate(self._feature_names):
            if name.startswith("rssi_") and not name.startswith("rssi_mean_floor_") and not name.startswith("rssi_best_floor") and not name.startswith("rssi_gap_"):
                # Smoothed RSSI per anchor
                aid = name[len("rssi_"):]
                vec[i] = smoothed.get(aid, MISSING_RSSI_SENTINEL)
            elif name.startswith("rssi_mean_floor_"):
                vec[i] = features.get(name, MISSING_RSSI_SENTINEL)
            elif name in features:
                vec[i] = features[name]
            else:
                # Default: 0 for deltas, variances, ranks, counts
                vec[i] = 0.0

        return vec

    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(
        X: np.ndarray,
        y: np.ndarray,
        anchor_count: int,
        factor: int = 5,
        rssi_noise_std: float = 1.5,
        dropout_prob: float = 0.15,
        balance_classes: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment training data with RSSI jitter and anchor dropout.

        Parameters
        ----------
        X : (N, D) feature matrix.
        y : (N,) label array.
        anchor_count : number of per-anchor RSSI columns at the start of X.
        factor : augmentation multiplier per sample.
        rssi_noise_std : std-dev of Gaussian noise added to RSSI columns.
        dropout_prob : probability of masking each anchor per augmented sample.
        balance_classes : if True, oversample minority classes so each class
            has ``max_class_count * (factor + 1)`` total samples.
        rng : numpy random generator (for reproducibility).

        Returns
        -------
        X_aug, y_aug : augmented arrays (original data included).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        aug_X_parts = [X]
        aug_y_parts = [y]

        for _ in range(factor):
            X_copy = X.copy()

            # Add Gaussian noise to the first `anchor_count` columns (RSSI)
            noise = rng.normal(0, rssi_noise_std, size=(X.shape[0], anchor_count))
            # Only add noise to non-sentinel values
            mask = X_copy[:, :anchor_count] > MISSING_RSSI_SENTINEL
            X_copy[:, :anchor_count] += noise * mask

            # Random anchor dropout: set some RSSI columns to sentinel
            dropout_mask = rng.random((X.shape[0], anchor_count)) < dropout_prob
            X_copy[:, :anchor_count] = np.where(
                dropout_mask, MISSING_RSSI_SENTINEL, X_copy[:, :anchor_count]
            )

            aug_X_parts.append(X_copy)
            aug_y_parts.append(y.copy())

        X_aug = np.vstack(aug_X_parts)
        y_aug = np.concatenate(aug_y_parts)

        # Class balancing: oversample minority classes to match majority
        if balance_classes:
            classes, counts = np.unique(y_aug, return_counts=True)
            max_count = counts.max()
            extra_X, extra_y = [], []
            for cls, cnt in zip(classes, counts):
                if cnt >= max_count:
                    continue
                deficit = max_count - cnt
                cls_indices = np.where(y_aug == cls)[0]
                # Sample with replacement from existing augmented pool
                chosen = rng.choice(cls_indices, size=deficit, replace=True)
                noisy = X_aug[chosen].copy()
                noise = rng.normal(0, rssi_noise_std, size=(deficit, anchor_count))
                valid = noisy[:, :anchor_count] > MISSING_RSSI_SENTINEL
                noisy[:, :anchor_count] += noise * valid
                extra_X.append(noisy)
                extra_y.append(np.full(deficit, cls))
            if extra_X:
                X_aug = np.vstack([X_aug] + extra_X)
                y_aug = np.concatenate([y_aug] + extra_y)

        return X_aug, y_aug

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        samples: list[dict],
        augment_factor: int = 5,
        n_estimators: int = 200,
        cv_folds: int = 5,
        rssi_noise_std: float = 1.5,
    ) -> dict:
        """Train on fingerprint sample dicts from the database.

        Each sample dict must contain at minimum:
        - ``"zone_label"`` (str): sub-zone name (e.g. ``"office_dog_bed"``)
        - ``"room"`` (str): parent room name (e.g. ``"office"``)
        - ``"floor"`` (int): floor number
        - ``"rssi_vector"`` (dict): ``{anchor_id: mean_rssi, ...}``

        For backward compatibility, if ``"zone_label"`` is missing, falls
        back to ``"location_label"``.  If ``"room"`` is missing, the zone
        label is used as the room name (degenerate: zone == room).

        Optionally:
        - ``"features"`` (dict): pre-computed FeatureEngine output.
          If absent, only smoothed RSSI + zero-filled derived features
          are used (adequate for initial training).

        Parameters
        ----------
        samples : list[dict]
            Rows from ``fingerprint_samples`` table.
        augment_factor : int
            Data augmentation multiplier (0 = no augmentation).
        n_estimators : int
            Number of trees in the forest.
        cv_folds : int
            Stratified K-fold count for cross-validation.

        Returns
        -------
        dict
            Metrics: accuracy, macro_f1, per_class_f1, confusion_matrix,
            cv_scores, top_features.
        """
        if len(samples) < 2:
            raise ValueError("Need at least 2 samples to train")

        # -- Build X, y --------------------------------------------------------
        X_rows = []
        y_labels = []
        zone_to_room: dict[str, str] = {}

        for sample in samples:
            # Sub-zone label (falls back to location_label for compat)
            zone = sample.get("zone_label", sample.get("location_label", "unknown"))
            room = sample.get("room", zone)
            zone_to_room[zone] = room

            rssi_vec = sample.get("rssi_vector", {})
            feat_dict = sample.get("features", {})

            vec = self._assemble_vector(feat_dict, smoothed_rssi=rssi_vec)
            X_rows.append(vec)
            y_labels.append(zone)

        self._zone_to_room = zone_to_room

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_labels)

        # -- Augmentation ------------------------------------------------------
        anchor_count = len(self._anchor_ids)
        if augment_factor > 0:
            X, y = self._augment(X, y, anchor_count, factor=augment_factor,
                                 rssi_noise_std=rssi_noise_std)

        logger.info(
            "Training RF: %d samples (%d original × %d aug), %d features, %d classes",
            len(y),
            len(samples),
            augment_factor + 1,
            X.shape[1],
            len(np.unique(y)),
        )

        # -- Fit model ---------------------------------------------------------
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=3,
            min_samples_split=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        self._classes = self._model.classes_

        # -- Cross-validation metrics -----------------------------------------
        metrics = self._evaluate_cv(X, y, cv_folds=cv_folds)

        logger.info(
            "RF trained — accuracy=%.3f  macro_f1=%.3f  classes=%s",
            metrics["accuracy"],
            metrics["macro_f1"],
            list(self._classes),
        )
        return metrics

    def _evaluate_cv(self, X: np.ndarray, y: np.ndarray, cv_folds: int) -> dict:
        """Run stratified k-fold cross-validation and compile metrics."""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = class_counts.min()

        # Adjust folds if any class has fewer samples than requested folds
        effective_folds = min(cv_folds, min_class_count)
        if effective_folds < 2:
            # Not enough data for CV — report training metrics only
            y_pred = self._model.predict(X)
            return self._compile_metrics(y, y_pred, cv_scores=[])

        skf = StratifiedKFold(
            n_splits=effective_folds, shuffle=True, random_state=42
        )
        y_pred = cross_val_predict(self._model, X, y, cv=skf)
        cv_scores = []
        for train_idx, test_idx in skf.split(X, y):
            fold_pred = self._model.predict(X[test_idx])
            cv_scores.append(float(accuracy_score(y[test_idx], fold_pred)))

        return self._compile_metrics(y, y_pred, cv_scores=cv_scores)

    def _compile_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, cv_scores: list[float]
    ) -> dict:
        """Build the metrics dict stored in model_versions."""
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        per_class = f1_score(y_true, y_pred, labels=labels, average=None)
        per_class_f1 = {lbl: round(float(f), 4) for lbl, f in zip(labels, per_class)}

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Feature importances
        importances = {}
        if self._model is not None:
            for name, imp in zip(self._feature_names, self._model.feature_importances_):
                importances[name] = round(float(imp), 5)
        top_features = dict(
            sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
        )

        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(
                float(f1_score(y_true, y_pred, average="macro")), 4
            ),
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.tolist(),
            "confusion_labels": labels,
            "cv_scores": [round(s, 4) for s in cv_scores],
            "top_features": top_features,
            "n_classes": len(labels),
            "n_features": len(self._feature_names),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features: dict,
        smoothed_rssi: Optional[dict] = None,
    ) -> tuple[str, float, dict]:
        """Predict zone label from a feature dict (all zones).

        Parameters
        ----------
        features : dict
            Output of ``FeatureEngine.update()``.
        smoothed_rssi : dict | None
            anchor_id → smoothed RSSI (for the ``rssi_<anchor>`` features).

        Returns
        -------
        label : str
            Zone label, e.g. ``"office_dog_bed"`` or ``"kitchen"``.
        confidence : float
            Probability of the predicted class (0–1).
        probabilities : dict
            ``{label: probability}`` for all classes.
        """
        if self._model is None:
            raise RuntimeError("No model loaded — call train() or load() first")

        vec = self._assemble_vector(features, smoothed_rssi).reshape(1, -1)
        proba = self._model.predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        label = str(self._classes[idx])
        confidence = float(proba[idx])
        probabilities = {
            str(cls): round(float(p), 4)
            for cls, p in zip(self._classes, proba)
        }
        return label, confidence, probabilities

    def predict_for_room(
        self,
        features: dict,
        smoothed_rssi: Optional[dict] = None,
        room: str = "",
    ) -> tuple[Optional[str], float, dict]:
        """Predict sub-zone scoped to a specific room.

        Filters RF probabilities to only zones belonging to the given
        room (via ``zone_to_room`` mapping), then renormalizes.  This
        is the primary prediction method for hierarchical fusion where
        the particle filter determines the room.

        Parameters
        ----------
        features : dict
            Output of ``FeatureEngine.update()``.
        smoothed_rssi : dict | None
            anchor_id → smoothed RSSI.
        room : str
            Parent room name from the particle filter's polygon lookup
            (e.g. ``"kitchen"``, ``"office"``).

        Returns
        -------
        label : str | None
            Best zone within the room, or ``None`` if no zones match.
        confidence : float
            Renormalized probability of the best zone (0–1).
        probabilities : dict
            ``{zone: probability}`` for zones in this room (renormalized).
        """
        _, _, all_probs = self.predict(features, smoothed_rssi)

        # Filter to zones belonging to this room
        scoped = {
            z: p for z, p in all_probs.items()
            if self._zone_to_room.get(z) == room
        }
        if not scoped:
            return None, 0.0, {}

        # Renormalise
        total = sum(scoped.values())
        if total > 0:
            scoped = {z: round(p / total, 4) for z, p in scoped.items()}

        best = max(scoped, key=scoped.get)
        return best, scoped[best], scoped

    @property
    def is_trained(self) -> bool:
        """Whether a model is loaded and ready for prediction."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def zone_to_room(self) -> dict[str, str]:
        """Zone label → parent room name mapping."""
        return dict(self._zone_to_room)

    def save(self, path: str) -> None:
        """Serialize trained model + metadata to a joblib file."""
        if self._model is None:
            raise RuntimeError("No model to save — call train() first")
        artifact = {
            "model": self._model,
            "feature_names": self._feature_names,
            "anchor_ids": self._anchor_ids,
            "classes": self._classes.tolist(),
            "zone_to_room": self._zone_to_room,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        logger.info("Saved RF model to %s", path)

    def load(self, path: str) -> None:
        """Deserialize a model from a joblib file."""
        artifact = joblib.load(path)
        self._model = artifact["model"]
        self._feature_names = artifact["feature_names"]
        self._anchor_ids = artifact["anchor_ids"]
        self._classes = np.array(artifact["classes"])
        # zone_to_room introduced in sub-zone refactor; compat with older models
        self._zone_to_room = artifact.get("zone_to_room", {})
        if not self._zone_to_room:
            # Legacy model: each class label IS the room (no sub-zones)
            for cls in self._classes:
                self._zone_to_room[str(cls)] = str(cls)
        logger.info(
            "Loaded RF model from %s (%d classes, %d features, %d rooms)",
            path,
            len(self._classes),
            len(self._feature_names),
            len(set(self._zone_to_room.values())),
        )

    @classmethod
    def from_file(cls, path: str) -> "ZoneClassifier":
        """Factory: create a ZoneClassifier from a saved model file."""
        artifact = joblib.load(path)
        instance = cls(anchor_ids=artifact["anchor_ids"])
        instance._model = artifact["model"]
        instance._feature_names = artifact["feature_names"]
        instance._classes = np.array(artifact["classes"])
        instance._zone_to_room = artifact.get("zone_to_room", {})
        if not instance._zone_to_room:
            for c in instance._classes:
                instance._zone_to_room[str(c)] = str(c)
        return instance

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names the model expects."""
        return list(self._feature_names)

    @property
    def feature_importances(self) -> dict[str, float]:
        """Feature name → Gini importance (requires trained model)."""
        if self._model is None:
            return {}
        return {
            name: round(float(imp), 5)
            for name, imp in zip(
                self._feature_names, self._model.feature_importances_
            )
        }

    @property
    def classes(self) -> list[str]:
        """List of zone class labels."""
        if self._classes is None:
            return []
        return [str(c) for c in self._classes]

    @property
    def anchor_ids(self) -> list[str]:
        """Anchor IDs used for feature assembly."""
        return list(self._anchor_ids)

    def __repr__(self) -> str:
        status = "trained" if self._model is not None else "untrained"
        n_cls = len(self._classes) if self._classes is not None else 0
        n_rooms = len(set(self._zone_to_room.values())) if self._zone_to_room else 0
        return (
            f"ZoneClassifier({status}, anchors={len(self._anchor_ids)}, "
            f"features={len(self._feature_names)}, zones={n_cls}, rooms={n_rooms})"
        )


# Backwards-compatible alias
RoomClassifier = ZoneClassifier
