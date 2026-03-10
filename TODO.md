# Project Todo List

## Phase 2A: Core Algorithms (no hardware needed)

- [X] **1. Kalman filter module** — `services/inference/filters/kalman.py`. Per-anchor RSSI smoother. Test against real InfluxDB data from single anchor.
  - Hardware: None
  - Dependencies: —

- [X] **2. Occupancy grid generator** — Parse `layout.json` polygons into walkable/blocked bitmap. Particle filter needs this for wall constraints.
  - Hardware: None
  - Dependencies: —

- [ ] **3. Feature engineering pipeline** — Delta RSSI, anchor rankings, rolling variance, cross-floor attenuation features. Build as standalone module.
  - Hardware: None
  - Dependencies: —

- [X] **4. Particle filter** — `services/inference/filters/particle.py`. Core Bayesian engine with motion model (~1 m/s dog speed), wall constraints from occupancy grid, resampling. Test with `simulate_mqtt.py`.
  - Hardware: None
  - Dependencies: #2

- [X] **5. Floor transition HMM** — Stairway-aware floor change logic. Prevent impossible transitions (Floor 1→3 skip). Purely geometric/probabilistic.
  - Hardware: None
  - Dependencies: —

- [ ] **6. Random Forest classifier skeleton** — `services/inference/models/classifier.py`. Training + prediction interface, feature pipeline integration. Can't train yet but can build scaffolding.
  - Hardware: None
  - Dependencies: #3

- [X] **7. PostgreSQL integration** — Wire up `psycopg2`/`sqlalchemy` in inference service to write to `position_history` and read from `fingerprint_samples`. Schema already exists in `init.sql`.
  - Hardware: None
  - Dependencies: —

- [ ] **8. Site survey collection script** — `scripts/site_survey.py`. CLI tool to record labeled RSSI vectors at grid positions, store in PostgreSQL `fingerprint_samples` table. Build now, use later.
  - Hardware: None (to write)
  - Dependencies: #7

- [X] **9. Integrate Kalman + Particle into `run_inference()`** — Replace strongest-anchor stub with real pipeline. Test end-to-end with simulator.
  - Hardware: None
  - Dependencies: #1, #4, #5

## Phase 2B: Dashboard Enhancements (no hardware needed)

- [ ] **10. Particle cloud visualization** — Show particle distribution on floor plan.
  - Hardware: None
  - Dependencies: #4

- [ ] **11. Position history trail** — Draw recent position path on floor plan.
  - Hardware: None
  - Dependencies: —

- [ ] **12. 24-hour residence heatmap** — Query InfluxDB/Postgres for historical dwell time.
  - Hardware: None
  - Dependencies: #7

- [ ] **13. Activity inference display** — Sleeping/Moving/Stationary from rolling variance.
  - Hardware: None
  - Dependencies: #3

## Phase 3: Hardware Deployment

- [ ] **14. Purchase + flash remaining AtomS3 Lites** (9 units)
  - Hardware: **9× AtomS3 Lite**
  - Dependencies: —

- [ ] **15. Deploy anchors across all 3 floors** — Mount, configure WiFi/MQTT/room names per `layout.json` anchor IDs.
  - Hardware: **All AtomS3 Lites**
  - Dependencies: #14

- [ ] **16. Update `layout.json` anchors** — Add real positions for all deployed anchors (measure from room corners).
  - Hardware: **All anchors mounted**
  - Dependencies: #15

- [ ] **17. Remove Floor 2 placeholder anchors** from `layout.json` — Replace with real positions.
  - Hardware: **Floor 2 anchors deployed**
  - Dependencies: #15

## Phase 4: Data Collection (all anchors required)

- [ ] **18. Multi-anchor smoke test** — Verify all 10 anchors report RSSI for the beacon simultaneously.
  - Hardware: **All anchors + beacon**
  - Dependencies: #15

- [ ] **19. Execute site survey** — Walk grid with beacon, run `site_survey.py` at each cell (60s per cell, rotate collar 360°).
  - Hardware: **All anchors + beacon**
  - Dependencies: #8, #18

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
