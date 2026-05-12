# Problems & Fixes Log

A running record of issues encountered during development of the Bayesian Pet Localization system, their root-cause analysis, and the solutions applied. Intended as a reference for the development blog and project portfolio.

---

## Entry 5 — Zone Classifier Low F1 Score (0.62) Across 22 POI-Based Classes

**Date:** 2025-04-14
**Severity:** Medium — zone-likelihood particle reweighting was operating on unreliable class probabilities

### Symptom

After implementing POI-based sub-zone classification (22 classes derived from 14 POIs + room-level fallbacks), the trained Random Forest achieved a cross-validated accuracy of 96.1 % but a **macro F1 of only 0.616** on the hold-out evaluation. Several POI zones had F1 scores in the 0.30–0.50 range:

| Zone | F1 |
|---|---|
| `master_bed_on_master_bed` | 0.45 |
| `master_bath` | 0.56 |
| `living_room` (catch-all) | 0.71 |
| `master_bed_dog_bed` | 0.72 |

Because the zone-likelihood particle reweighting step (`ZONE_LIKELIHOOD_STRENGTH = 0.3`) directly uses the classifier's per-class probabilities, low-confidence predictions were injecting noisy reweighting into the particle filter.

### Root-Cause Analysis

Three compounding factors were identified:

#### 1. Excessive Augmentation Noise (σ = 3.0 dBm)

The `_augment()` method in `classifier.py` added Gaussian noise with `rssi_noise_std = 3.0` dBm to generate synthetic training samples. Real BLE RSSI variation at a stationary point is typically 1–2 dBm. At 3.0 dBm, augmented samples from adjacent POI zones overlapped significantly in feature space, making the classifier unable to distinguish nearby zones within the same room.

#### 2. Class-Imbalanced Training Data

The 130 survey samples were unevenly distributed across zones — some POIs had as few as 2 samples (e.g., `master_bath`) while `living_kitchen` had 45. After POI relabeling, popular zones dominated the augmented dataset. Even though the Random Forest used `class_weight="balanced"` internally, the augmentation step produced far more synthetic samples for majority classes, diluting minority class representation in the training set.

#### 3. POI Radius Too Large (3.0 ft)

The `poi_radius_ft = 3.0` setting in `layout.json` determined which survey samples were relabeled from their room-level zone to a POI-specific zone. At 3.0 ft, some edge-case samples were assigned to POI zones even though their RSSI fingerprint was ambiguous between the POI and the room-level class, introducing label noise.

### Fixes Applied

#### Fix 1 — Reduced Augmentation Noise (`classifier.py`)

Changed the default `rssi_noise_std` from **3.0 → 1.5** dBm in `_augment()`. Added a `rssi_noise_std` parameter to `train()` so the value can be tuned from the CLI without modifying source. The training script gained a `--rssi-noise-std` argument (default 1.5).

#### Fix 2 — Class-Balanced Augmentation (`classifier.py`)

Added a `balance_classes=True` parameter to `_augment()`. When enabled, after initial augmentation the method identifies the majority class count and oversamples minority classes by generating additional noise-augmented copies until all classes have equal representation. This complements the Random Forest's internal `class_weight="balanced"` with data-level balancing.

#### Fix 3 — Tightened POI Radius (`layout.json`)

Reduced `poi_radius_ft` from **3.0 → 2.5** ft. This dropped 5 ambiguous edge-case samples from POI relabeling (55 → 50 relabeled), reducing label noise while still capturing the core fingerprint of each POI zone.

### Results

Three models were trained for comparison (all with `--augment-factor 8`, 22 classes):

| Configuration | Accuracy | Macro F1 | Samples |
|---|---|---|---|
| Baseline (σ=3.0, no balance, r=3.0ft) | 0.623 | 0.616 | 1,170 |
| σ=1.5, balanced, r=3.0ft | 0.919 | 0.916 | 3,762 |
| **σ=1.5, balanced, r=2.5ft** | **0.926** | **0.924** | **4,158** |

Per-class F1 improvements (baseline → final):

| Zone | Before | After |
|---|---|---|
| `master_bed_dog_bed` | 0.72 | **0.99** |
| `master_bed_on_master_bed` | 0.45 | **0.95** |
| `master_bath` | 0.56 | **0.96** |
| `living_room` (catch-all) | 0.71 | **0.79** |
| `office_dog_bed` | 0.46 | **0.90** |
| `hallway` | 0.61 | **0.81** |

The remaining weakest class is `living_room` at 0.79 — expected because it is the catch-all for the large open-plan living/kitchen area for points not near any POI.

### Verification

All 251 tests pass (5 skipped). Final model saved as `random_forest_v20260414_160036.joblib`.

---

## Entry 4 — Activity Detection Stuck on "Sleeping" / "Stationary"

**Date:** 2025-04-07
**Severity:** Medium — activity label never reflected real movement

### Symptom

The dashboard's Activity metric reported "sleeping" or "stationary" virtually 100 % of the time, even when the pet was visibly walking between rooms. The activity score (0–1) hovered near 0.0 regardless of movement.

### Root-Cause Analysis

Live sampling via `curl /position` confirmed the activity score was consistently below the sleeping threshold (0.15). Three compounding issues were identified:

#### 1. Delta Features Computed on Kalman-Smoothed RSSI

`_delta_features()` computed per-anchor RSSI change from the **Kalman-smoothed** signal. The Kalman filter (process_noise=0.5, measurement_noise=1.0) aggressively smooths step-to-step variation. Real movement-induced RSSI jumps of 3–5 dBm in the raw signal appeared as only 0.1–0.3 dBm deltas in the smoothed stream — well below any useful threshold.

Live data confirmed: 5 consecutive inference cycles showed smoothed RSSI changing by only 0.1–0.3 dBm between steps, while raw RSSI showed 1–3 dBm variation.

#### 2. Normalization Ceilings Far Too High

The activity score normalized variance and delta components against generous ceilings:

| Component | Ceiling | Typical Real Value | Effective Utilization |
|---|---|---|---|
| `VARIANCE_NORMALISATION_MAX` | 20.0 dBm² | 0.5–2.0 dBm² | 2–10 % |
| `DELTA_NORMALISATION_MAX` | 5.0 dBm | 0.1–0.3 dBm (smoothed) | 2–6 % |

Even when the pet was actively walking, the normalized components produced activity scores of 0.02–0.08 — deep in "sleeping" territory.

#### 3. No Position-Based Movement Signal

The activity score relied entirely on RSSI statistics (variance + delta), ignoring the most direct evidence of movement: the particle filter's own (x, y, floor) position estimate. A pet walking across a room produces clear position displacement even when RSSI changes are noisy or ambiguous.

### Fixes Applied

#### Fix 1 — Raw RSSI Deltas (`features.py`)

`_delta_features()` now computes deltas from **raw** RSSI values instead of Kalman-smoothed. This preserves the full magnitude of movement-induced signal changes. A new `_prev_raw` dict tracks the previous step's raw readings alongside the existing `_prev_smoothed`.

#### Fix 2 — Position Displacement as Primary Activity Signal (`features.py`, `main.py`)

Added `_compute_displacement()` which maintains a rolling buffer of particle filter `(x, y, floor)` positions (`POSITION_HISTORY_SIZE = 10`). Each inference cycle, the Euclidean distance from the oldest to newest buffered position is computed. Cross-floor transitions return a fixed 1.0 ft to indicate movement without a misleading 2D distance.

The `update()` method now accepts an optional `position` parameter. In `main.py`, the pipeline was reordered so the feature engine runs **after** the particle filter step, allowing it to receive the current position estimate:

```python
estimate = particle_filter.step(smoothed_rssi, dt)
position = (estimate["x"], estimate["y"], estimate["floor"])
computed_features = feature_engine.update(rssi_vector, smoothed_rssi, now, position=position)
```

#### Fix 3 — Lowered Normalization Ceilings (`features.py`)

Reduced ceilings to match real BLE dynamics:

| Parameter | Before | After |
|---|---|---|
| `VARIANCE_NORMALISATION_MAX` | 20.0 dBm² | **8.0 dBm²** |
| `DELTA_NORMALISATION_MAX` | 5.0 dBm | **3.0 dBm** |

#### Fix 4 — Rebalanced Activity Score Weights (`features.py`)

The composite activity score now blends three signals instead of two:

| Component | Old Weight | New Weight |
|---|---|---|
| RSSI variance | 60 % | **25 %** |
| RSSI delta (raw) | 40 % | **25 %** |
| Position displacement | — | **50 %** |

Displacement is weighted highest because it is the most direct measure of movement and is unaffected by RSSI noise floors.

A new normalization ceiling `DISPLACEMENT_NORMALISATION_MAX = 4.0 ft` maps walking-speed displacement to the upper end of the 0–1 range.

#### Fix 5 — Increased Feature Window (`main.py`)

`FEATURE_WINDOW_SIZE` increased from 10 to **20** samples, providing a longer rolling window for more stable variance estimates and displacement measurement.

### Verification

All 251 tests pass (5 skipped). 7 new tests added for displacement computation: stationary, Pythagorean distance, cross-floor, history windowing, and displacement-driven score.

---

## Entry 3 — Stale Retained MQTT Messages After Anchor Rename

**Date:** 2025-04-02
**Severity:** Low — caused false readings in diagnostic queries only

### Symptom

After renaming anchor `3F_Wifes_Office` → `3F_Office` in the ESPresense web UI, MQTT wildcard subscriptions (`espresense/rooms/+/max_distance`) returned **9 results instead of 8** — the old `3f_wifes_office` topic still responded with stale values (e.g., `max_distance: 16`) alongside the live `3f_office` topic. The inference pipeline was unaffected because it had already been updated to subscribe to `3f_office`, but diagnostic monitoring was misleading.

### Root Cause

Mosquitto retains the last message on any topic published with the retain flag. When ESPresense renames a node, it starts publishing on the new topic but never clears the old one. The broker holds the old retained messages indefinitely.

### Fix

Published empty retained messages to all known sub-topics of the old name:

```bash
for topic in status name max_distance absorption tx_ref_rssi rx_adj_rssi ...
    mosquitto_pub -t "espresense/rooms/3f_wifes_office/$topic" -r -n
end
```

The `-r -n` flags publish a zero-length retained message, which Mosquitto interprets as "delete the retained message on this topic."

**Lesson:** When renaming ESPresense nodes, always clear stale retained MQTT topics on the broker.

---

## Entry 2 — Silent Anchor Data Loss from ESPresense max_distance Filter

**Date:** 2025-04-02
**Severity:** High — 3 of 8 anchors silently dropped from the pipeline

### Symptoms

1. With the beacon on Floor 1, the inference pipeline consistently received data from only 5 of 8 anchors. The 3 Floor-3 anchors — `3f_hallway`, `3f_office`, and `3f_master_bed` — were completely absent (0 messages) or barely reporting (41 out of 1000 messages). The particle filter and Floor HMM were making floor-discrimination decisions with incomplete data.

2. The dashboard exhibited inconsistent latency and stuttering — some 1-second poll cycles returned fresh data while others showed no change, because the combined MQTT rate (~1.1 msgs/s across 5 anchors with 5s reporting throttle) meant most poll intervals had no new observations for several anchors.

### Root Cause

ESPresense's default `max_distance` setting is **16 meters** (~52 ft). Any beacon whose estimated distance exceeds this threshold is silently dropped — the anchor sees the beacon in its BLE scan but does not publish it to MQTT. In a 3-story townhouse, the straight-line distance from Floor 1 to Floor 3 anchors is 15–21 ft, but ESPresense's distance estimate includes absorption/reflection factors that can inflate it beyond 16m (especially cross-floor).

The setting is documented as "Maximum distance to report (in meters) — if the distance is over this value (default 16m), it's likely inaccurate and not worth including in trilateration." For single-floor deployments this is reasonable, but for multi-floor BLE localization it silently destroys cross-floor observations.

### Fix

Changed `max_distance` from **16 → 40** on all 8 anchors via the ESPresense web UI (Settings → Filtering). Additionally changed `skip_reporting` from **5000ms → 2000ms** to increase per-anchor reporting rate, and added the beacon's iBeacon UUID to the **include-only filter** to eliminate noise from non-beacon BLE devices.

**Results after fix:**

| Metric | Before | After |
|---|---|---|
| Active anchors | 5 of 8 | **8 of 8** |
| Overall MQTT rate | 1.1 msgs/s | **8.0 msgs/s** |
| Per-anchor median gap | 5.0s | **~1.0s** |
| 3f_hallway messages | 0 | 70 per sample |
| 3f_office messages | 0 | 58 per sample |

The particle filter now receives observations from all 8 anchors every inference cycle, significantly improving floor discrimination and multilateration accuracy. Dashboard latency also improved from inconsistent to near-real-time.

---

## Entry 1 — Particle Filter Stuck on Wrong Floor & Unable to Navigate Walls

**Date:** 2025-07-16
**Severity:** Critical — core tracking completely non-functional in multi-floor scenarios

### Symptoms

1. **Floor transition failure.** With the iBeacon physically carried from the 1F Office to the 2F Living area, the dashboard's particle-filter plot remained locked on Floor 1 with particles clustered at the top of the staircase. The Floor HMM correctly shifted belief toward Floor 2, but the particle cloud never followed.

2. **Wall-stuck particles.** When walking from the staircase through the hallway to the office on Floor 1, particles became trapped on the staircase side of the hallway–office wall. They never navigated around the wall through the doorway (2.29 ft wide) to reach the office.

### Root-Cause Analysis

A full data-flow analysis from MQTT ingestion through the inference pipeline revealed **four compounding issues**:

#### 1. Inference Over-Triggering (~8 Hz)

The MQTT `on_message` callback invoked `run_inference()` on every received message. ESPresense publishes a message per anchor per scan cycle—with 8 anchors this produced roughly 8 inference steps per second. Each step's `dt ≈ 0.125 s` yielded:

| Parameter | Value |
|---|---|
| `max_displacement` (`MAX_SPEED_FT_S × dt`) | 0.41 ft |
| `sigma` (`DIFFUSION_SIGMA × √dt`) | 0.53 ft |

At these tiny step sizes, particles could barely move each tick. Navigating from the staircase through the hallway and around the wall to the 2.29 ft doorway was nearly impossible by random diffusion alone—it required many consecutive lucky draws in the right direction.

#### 2. Floor Transition Probability Not Scaled by `dt`

`_maybe_transition_floors()` computed:

```python
p_transition = min(dest_belief * 0.5, 0.4)
```

This probability was applied every step regardless of how much time had elapsed. At 8 Hz, the effective per-second transition rate was `p × 8`, far higher than intended when the HMM signals a transition—but paradoxically still ineffective because the small number of particles near stairs would transition and immediately get resampled away.

#### 3. No Floor Teleport Mechanism

When the Floor HMM accumulated strong evidence (e.g., 90 % belief on Floor 2) but particles were physically stuck on Floor 1 behind walls, there was no recovery path. The only way to change floors was for particles to random-walk to a stairway entry point—which was itself blocked by the wall-navigation problem.

#### 4. Pure Random Walk (No Directed Motion)

`predict()` used isotropic Gaussian diffusion:

```python
self._x += rng.normal(0, sigma, n)
self._y += rng.normal(0, sigma, n)
```

Particles explored uniformly in all directions with no bias toward areas of stronger RSSI signal. In a narrow hallway with a small doorway, the vast majority of random steps either hit a wall (and were clamped) or moved away from the doorway. There was no mechanism to pull particles toward regions consistent with the observations.

### Fixes Applied

#### Fix 1 — Inference Rate Throttling (`main.py`)

Added a minimum interval between inference cycles:

```python
INFERENCE_MIN_INTERVAL = 0.25  # seconds → max 4 Hz
```

`on_message()` now checks `time.monotonic()` against the last inference timestamp and skips if the interval has not elapsed. RSSI readings are still buffered (latest-per-anchor), so no data is lost. The dashboard polls independently at its own refresh rate and is unaffected.

**Effect:** `dt ≈ 0.25 s`, `max_displacement ≈ 0.82 ft`, `sigma ≈ 0.75 ft`—particles can now traverse a doorway in a handful of steps rather than dozens.

#### Fix 2 — dt-Scaled Floor Transition Rate (`particle.py`)

Replaced the fixed probability with a rate-based formulation:

```python
FLOOR_TRANSITION_RATE_HMM = 0.8  # transitions/sec at full HMM belief

p_transition = min(dest_belief * FLOOR_TRANSITION_RATE_HMM * dt, 0.4)
```

Transition probability is now proportional to elapsed time, making behavior consistent across different inference frequencies.

#### Fix 3 — Floor Teleport on Sustained HMM Disagreement (`particle.py`)

New method `_maybe_teleport()` checks whether the HMM's most-likely floor differs from the particle majority floor. If the HMM belief exceeds `TELEPORT_BELIEF_THRESHOLD` (0.85) for at least `TELEPORT_HOLDOFF_SEC` (2.0 s) of cumulative disagreement, the lowest-weight `TELEPORT_FRACTION` (30 %) of particles are redistributed uniformly across walkable cells on the target floor.

This provides a recovery path when particles are stuck on the wrong floor and cannot reach a stairway entry. The holdoff period prevents spurious jumps from brief RSSI fluctuations, and the 30 % fraction preserves particle diversity on the current floor in case the HMM is momentarily wrong.

#### Fix 4 — RSSI-Gradient Drift Bias (`particle.py`)

New method `_compute_drift_targets()` computes a per-floor weighted centroid of anchor positions, using linearized RSSI as weights:

```python
power = 10 ** ((rssi + 100) / 10)  # dBm → linear (shifted to stay positive)
weighted centroid per floor
```

In `predict()`, each particle receives a small deterministic pull toward the drift target on its floor:

```python
drift_scale = DRIFT_ALPHA * dt   # DRIFT_ALPHA = 0.15 /sec
self._x += drift_scale * (target_x - self._x)
self._y += drift_scale * (target_y - self._y)
```

This replaces fully isotropic diffusion with biased diffusion: particles still explore randomly, but are nudged toward the region of strongest signal evidence. This dramatically improves doorway navigation because particles are now pulled toward the office when office-side anchors report strong RSSI.

### Verification

All 225 existing tests pass after the changes (5 skipped, unrelated). Live testing pending Docker rebuild.

---

## Entry 0 — Occupancy Grid Wall Collisions Ignoring Doorways

**Date:** 2025-03 (initial development)
**Severity:** Medium — particles trapped in rooms

### Symptom

During early occupancy-grid integration, particles that should have been able to walk through doorways between rooms were being blocked. The `is_walkable()` check was rejecting doorway cells.

### Root Cause

Doorways in `layout.json` define passage openings between rooms, but the occupancy-grid rasterization was filling wall segments without carving out doorway regions. The grid treated doorway cells as walls.

### Fix

The `OccupancyGrid` construction was updated to explicitly mark doorway cells as walkable after wall rasterization. `_rasterize_walls()` now iterates over all doorways in the layout and clears (sets to `True`) any grid cells that fall within the doorway bounding box. This ensures connectivity between rooms.

---
