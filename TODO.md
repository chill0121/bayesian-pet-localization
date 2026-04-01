# Project Todo List

## Phase 2A: Core Algorithms (no hardware needed)

- [X] **1. Kalman filter module** — `services/inference/filters/kalman.py`. Per-anchor RSSI smoother. Test against real InfluxDB data from single anchor.
  - Hardware: None
  - Dependencies: —

- [X] **2. Occupancy grid generator** — Parse `layout.json` polygons into walkable/blocked bitmap. Particle filter needs this for wall constraints.
  - Hardware: None
  - Dependencies: —

- [X] **3. Feature engineering pipeline** — Delta RSSI, anchor rankings, rolling variance, cross-floor attenuation features. Build as standalone module.
  - Hardware: None
  - Dependencies: —

- [X] **4. Particle filter** — `services/inference/filters/particle.py`. Core Bayesian engine with motion model (~1 m/s dog speed), wall constraints from occupancy grid, resampling. Test with `simulate_mqtt.py`.
  - Hardware: None
  - Dependencies: #2

- [X] **5. Floor transition HMM** — Stairway-aware floor change logic. Prevent impossible transitions (Floor 1→3 skip). Purely geometric/probabilistic.
  - Hardware: None
  - Dependencies: —

- [X] **6. Random Forest classifier skeleton** — `services/inference/models/classifier.py`. Training + prediction interface, feature pipeline integration. Can't train yet but can build scaffolding.
  - Hardware: None
  - Dependencies: #3

- [X] **7. PostgreSQL integration** — Wire up `psycopg2`/`sqlalchemy` in inference service to write to `position_history` and read from `fingerprint_samples`. Schema already exists in `init.sql`.
  - Hardware: None
  - Dependencies: —

- [X] **8. Site survey collection script** — `scripts/site_survey.py`. CLI tool to record labeled RSSI vectors at grid positions, store in PostgreSQL `fingerprint_samples` table. Build now, use later.
  - Hardware: None (to write)
  - Dependencies: #7

- [X] **9. Integrate Kalman + Particle into `run_inference()`** — Replace strongest-anchor stub with real pipeline. Test end-to-end with simulator.
  - Hardware: None
  - Dependencies: #1, #4, #5

## Phase 2B: Dashboard Enhancements (no hardware needed)

- [X] **10. Particle cloud visualization** — Show particle distribution on floor plan.
  - Hardware: None
  - Dependencies: #4

- [X] **11. Position history trail** — Draw recent position path on floor plan.
  - Hardware: None
  - Dependencies: —

- [X] **12. 24-hour residence heatmap** — Query InfluxDB/Postgres for historical dwell time.
  - Hardware: None
  - Dependencies: #7

- [X] **13. Activity inference display** — Sleeping/Moving/Stationary from rolling variance.
  - Hardware: None
  - Dependencies: #3

## Phase 3: Hardware Deployment

- [X] **14. Purchase + flash remaining AtomS3 Lites** (9 units)
  - Hardware: **9× AtomS3 Lite**
  - Dependencies: —

- [X] **15. Deploy anchors across all 3 floors** — Mount, configure WiFi/MQTT/room names per `layout.json` anchor IDs.
  - Hardware: **All AtomS3 Lites**
  - Dependencies: #14

- [X] **16. Update `layout.json` anchors** — Add real positions for all deployed anchors (measure from room corners).
  - Hardware: **All anchors mounted**
  - Dependencies: #15

- [X] **17. Remove Floor 2 placeholder anchors** from `layout.json` — Replace with real positions.
  - Hardware: **Floor 2 anchors deployed**
  - Dependencies: #15

## Phase 4: Data Collection (all anchors required)

- [ ] **18. Multi-anchor smoke test** — Verify all anchors report RSSI for the beacon simultaneously.
  - Hardware: **All anchors + beacon**
  - Dependencies: #15

- [ ] **19. Execute site survey** — Walk grid with beacon, run `site_survey.py` at each cell (60s per cell, rotate collar 360°).
  - Hardware: **All anchors + beacon**
  - Dependencies: #8, #18

- [ ] **19.5 Use site survey data to calibrate inference & RSSI distance degradation (particle filter, constants, etc)** - After data collection we can verify wall attenuation calculations are accurate and calibrate particle filter as necessary. This should improve RSSI log loss calculations.
  - Hardware: **All anchors + beacon**
  - Dependencies: #8, #18, #19

- [ ] **19.75 Normalize Signal Strength Visualizations with real sampled ranges** - Currently they are almost always in the red and orange ranges which doesn't reflect reality, giving much less visual info.
  - Hardware: **All anchors + beacon**
  - Dependencies: #8, #18, #19

- [ ] **20. Labeled location collection** — Record RSSI at all POIs (couch, water bowl, dog beds, etc.) with extended dwell times.
  - Hardware: **All anchors + beacon**
  - Dependencies: #18

## Phase 5: Model Training (needs survey data)

- [ ] **21. Train Random Forest** on fingerprint data — Fit classifier, evaluate cross-validation accuracy.
  - Hardware: None (data from #19)
  - Dependencies: #6, #19, #20

- [ ] **22. Tune Kalman filter parameters** — Use multi-anchor data to optimize process/measurement noise.
  - Hardware: None (data from #19)
  - Dependencies: #1, #18

- [ ] **23. Tune particle filter** — Particle count, motion sigma, resampling threshold.
  - Hardware: None (data from #19)
  - Dependencies: #4, #21

- [ ] **24. Full pipeline integration test** — Live beacon tracking through all rooms, compare to ground truth.
  - Hardware: **All anchors + beacon**
  - Dependencies: #9, #21, #22, #23

## Phase 6: Polish (portfolio-ready)

- [ ] **25. Noise analysis notebook** — Raw vs Kalman-filtered RSSI plots, before/after comparison.
  - Hardware: None
  - Dependencies: #22

- [ ] **26. System architecture diagram** — MQTT data flow, ML pipeline visualization.
  - Hardware: None
  - Dependencies: —

- [ ] **27. Signal bleed analysis** — Document cross-floor RSSI attenuation patterns.
  - Hardware: **All anchors**
  - Dependencies: #19

- [ ] **28. Deploy to Raspberry Pi** — Move Docker stack from dev machine to Pi for permanent operation.
  - Hardware: **Raspberry Pi 4/5**
  - Dependencies: #24

## Phase 7: Post-Validation Improvements (after #24 baseline)

- [ ] **29. Hierarchical classifier** — If flat RF accuracy is poor on same-floor rooms, switch to two-stage: floor classifier → per-floor room classifier. Evaluate against flat model.
  - Hardware: None
  - Dependencies: #21, #24

- [ ] **30. RF → Particle fusion (Option B)** — Use RF per-room probability distribution as a likelihood term to re-weight particles, replacing simple confidence-threshold override. Tighter Bayesian integration.
  - Hardware: None
  - Dependencies: #21, #24

- [ ] **31. Anchor dropout indicators** — If classifier accuracy suffers from missing anchors, add `heard_<anchor>` binary features (0/1) per anchor to the feature vector. Gives RF explicit dropout signal.
  - Hardware: None
  - Dependencies: #21

- [ ] **32. Concept drift detection** — Monitor RF prediction vs. polygon label disagreement rate over time. Flag for retraining when disagreement exceeds threshold. Periodic re-survey cadence.
  - Hardware: None
  - Dependencies: #24

- [ ] **33. Hyperparameter grid search** — If baseline RF accuracy is <80%, sweep `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`. Use spatial CV to avoid overfitting to adjacent grid cells.
  - Hardware: None
  - Dependencies: #21

- [ ] **34. Path loss calibration** — Use fingerprint data to fit environment-specific `TX_POWER`, path loss exponent `N`, and `RSSI_SIGMA` per anchor. Improves particle filter position accuracy.
  - Hardware: None (data from #19)
  - Dependencies: #22, #19

- [ ] **35. Client-side particle interpolation** — Add a `clientside_callback` with a fast `dcc.Interval` (~33ms) that linearly interpolates particle positions between inference snapshots, decoupling visual framerate from the 250ms inference tick. Store prev/current snapshots in `dcc.Store`, lerp in JS.
  - Hardware: None
  - Dependencies: —

- [ ] **35. POI proximity labeling** — Use particle filter (x, y) estimate + known POI coordinates to detect sub-room locations (e.g. "on dog bed", "near water bowl") via distance threshold. Separate from room classifier.
  - Hardware: None
  - Dependencies: #20, #24

- [ ] **36. RPi 5 performance benchmarking** — Profile full inference pipeline on Raspberry Pi 5 (Cortex-A76 @ 2.4GHz) and Mac (Apple Silicon). Measure `step()` latency, wall-crossing overhead, and max sustainable MQTT throughput. If step time exceeds 125ms budget (8 msg/sec), vectorize `count_wall_crossings` with numpy or precompute wall-crossing lookup table.
  - Hardware: **Raspberry Pi 5**

- [ ] **37. Body-occlusion detector** — Detect when the pet's body is occluding the beacon (e.g. lying on collar). Occlusion analysis (2026-03-26) showed ~12 dB same-floor attenuation with anchor visibility dropping from 5→3. Three-layer defense: (1) **Detection** — flag occlusion when `n_reporting` drops AND same-floor anchors uniformly attenuate while cross-floor anchors are stable. Use `n_reporting`, per-anchor RSSI delta from baseline, and variance inversion (lower σ under occlusion) as indicators. (2) **Temporal smoothing** — already implemented via `ZoneSmoother` with activity-adaptive EMA. (3) **Training augmentation** — include occluded RSSI samples in `ZoneClassifier` training data so the RF learns to be robust to body attenuation patterns.
  - Hardware: None (captured calibration data in `scratch/occlusion_analysis_1F_Office_20260326.json`)
  - Dependencies: #21

- [ ] **38. Zone smoother (activity-adaptive EMA)** — Wire `ZoneSmoother` (`services/inference/models/zone_smoother.py`) into the inference pipeline after the `ZoneClassifier` prediction step. Applies an exponential moving average over sub-zone probability distributions to prevent zone-label snapping during body-occlusion or brief RSSI fluctuations. Alpha adapts to activity state: moving=0.3, stationary=0.05, sleeping=0.01. Resets on room change. Module and tests (`tests/test_zone_smoother.py`) are already written — just needs re-integration into `main.py:run_inference()`.
  - Hardware: None
  - Dependencies: #21