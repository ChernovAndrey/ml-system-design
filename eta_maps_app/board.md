# ML System Design — ETA Prediction for Maps App

## 1. Problem Statement
Predict the estimated time of arrival for **driving** trips in a maps application.

## 2. Requirements
- **Transport:** Driving only
- **Scale:** ~hundreds of requests/sec
- **Latency:** 100–300 ms acceptable
- **Data:** GPS traces (historical trips), real-time traffic, road network graph

## 3. Metrics

| Layer | Metric | Why |
|-------|--------|-----|
| Offline (primary) | MAPE (mean absolute % error) | Normalizes error by trip length |
| Offline (secondary) | Asymmetric loss (penalize underestimates 1.5–2x) | Late arrival is worse than early |
| Online | % trips within ±10% of predicted ETA | Intuitive for stakeholders |
| Business | User trust / re-engagement with navigation | Ultimate goal |

## 4. High-Level Architecture

```
User Query (origin, destination, departure_time)
        │
        ▼
┌───────────────────────┐
│   Routing Engine       │  ← road graph + real-time traffic
│   (Dijkstra / A*)      │     (e.g., OSRM / Valhalla)
│                        │
│   Output: ordered list │
│   of road segments     │
└──────────┬────────────┘
           │  route = [seg1, seg2, ..., segN]
           ▼
┌───────────────────────┐
│   Feature Engineering  │
│   (per-segment +       │
│    route-level)        │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────────────────────────────┐
│          HYBRID ETA MODEL                      │
│                                                │
│  ┌─────────────────┐                           │
│  │ Stage 1:        │                           │
│  │ Segment Model   │  per-segment features     │
│  │ (LightGBM)      │──→ t1, t2, ..., tN       │
│  └────────┬────────┘                           │
│           │  sum(t_i) = raw_ETA                │
│           ▼                                    │
│  ┌─────────────────┐                           │
│  │ Stage 2:        │  route-level features     │
│  │ Route Residual  │  + raw_ETA                │
│  │ Model (LightGBM)│──→ correction factor      │
│  └────────┬────────┘                           │
│           │                                    │
│           │  final_ETA = raw_ETA + correction  │
│           ▼                                    │
└───────────────────────────────────────────────┘
           │
           ▼
     ETA → User
```

## 5. Baseline
**Segment-speed summation:**
- For each segment: time = distance / speed_limit (or historical avg speed)
- ETA = Σ segment times
- Simple, no ML, decent on uncongested roads
- Fails on: traffic, turns, signals, time-of-day, weather

## 6. Features

### Per-Segment Features (Stage 1 input)
| Feature | Source | Why |
|---------|--------|-----|
| Segment length (m) | Road graph | Primary distance signal |
| Road type (highway/arterial/residential) | Road graph | Speed characteristics differ |
| Speed limit | Road graph | Upper bound on speed |
| Number of lanes | Road graph | Capacity proxy |
| Current traffic speed | Real-time traffic | Live congestion signal |
| Free-flow speed ratio | Historical + live | How congested vs normal |
| Historical avg speed (this hour, this DOW) | Historical trips | Time-pattern signal |
| Has traffic signal at end? | Road graph | Intersection delay |
| Turn angle to next segment | Road graph | Sharp turns = slowdown |
| Is on-ramp / off-ramp? | Road graph | Merge delays |

### Route-Level Features (Stage 2 input)
| Feature | Source | Why |
|---------|--------|-----|
| raw_ETA (from Stage 1) | Stage 1 output | Anchor prediction |
| Total distance | Route | Scale signal |
| Number of segments | Route | Complexity |
| % highway vs urban | Route | Trip character |
| Number of traffic signals | Route | Cumulative delay |
| Number of left turns | Route | Left turns are slow |
| Hour of day, day of week | Query time | Temporal patterns |
| Is holiday / special event | Calendar | Unusual traffic |
| Weather (rain/snow/clear) | Weather API | Road condition impact |
| # active incidents on route | Incident feed | Disruption signal |

## 7. Model Choice & Rationale

### Why LightGBM over Deep Learning (to start)?
| Consideration | LightGBM | Deep Learning |
|---------------|-----------|---------------|
| Tabular data performance | Excellent | Comparable but not better |
| Inference latency | ~1–5 ms | ~10–50 ms |
| Training speed | Fast | Slow |
| Interpretability | Feature importance, SHAP | Black box |
| Engineering complexity | Low | High (GPU serving) |

### When to consider Deep Learning?
- If segment **sequence interactions** matter a lot
- Graph Neural Networks on road network (à la Google DeepETA)
- Would pursue after baseline GBT proves the concept

## 8. Training Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    OFFLINE / TRAINING                         │
│                                                              │
│  Raw GPS Traces                                              │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────┐                                            │
│  │ Map Matching  │  HMM-based (Valhalla)                     │
│  │ GPS → road    │  snap noisy points to road segments       │
│  │ segments      │                                           │
│  └──────┬───────┘                                            │
│         ▼                                                    │
│  ┌──────────────┐     ┌───────────────────┐                  │
│  │ Per-Segment   │     │ Feature Store      │                 │
│  │ Ground Truth  │     │                   │                  │
│  │ Travel Times  │     │ Static: road type, │                 │
│  │ (from GPS     │     │   speed limit,     │                 │
│  │  timestamps)  │     │   lanes, signals   │                 │
│  └──────┬───────┘     │                   │                  │
│         │              │ Historical: avg    │                 │
│         │              │   speed by (seg,   │                 │
│         │              │   hour, DOW)       │                 │
│         │              │                   │                  │
│         │              │ Snapshot: traffic, │                 │
│         │              │   weather, events  │                 │
│         │              │   at trip time     │                 │
│         │              └────────┬──────────┘                  │
│         │                       │                             │
│         ▼                       ▼                             │
│  ┌──────────────────────────────────┐                        │
│  │  Join: labeled training examples  │                        │
│  │  (segment_features, actual_time)  │                        │
│  │  (route_features, actual_total)   │                        │
│  └──────────────┬───────────────────┘                        │
│                 │                                             │
│         ┌───────┴────────┐                                   │
│         ▼                ▼                                    │
│  ┌─────────────┐  ┌──────────────┐                           │
│  │ Train        │  │ Train         │                          │
│  │ Stage 1:     │  │ Stage 2:      │                          │
│  │ Segment      │  │ Route         │  ← uses Stage 1         │
│  │ Model        │  │ Residual      │    predictions as input  │
│  └──────┬──────┘  └──────┬───────┘                           │
│         │                │                                    │
│         ▼                ▼                                    │
│     Model Registry (versioned, A/B testable)                 │
│                                                              │
│  Split: TEMPORAL (train months 1–5, val month 6, test 7)     │
│  Retrain: weekly                                             │
│  Validation gate: new model must beat current on MAPE        │
└──────────────────────────────────────────────────────────────┘
```

## 9. Serving Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     ONLINE / SERVING                          │
│                                                              │
│  User Request                                                │
│       │                                                      │
│       ▼                          Latency budget:             │
│  ┌──────────┐                    ┌─────────────────────┐     │
│  │ Routing   │  ~50–100 ms       │ Routing:   50–100ms │     │
│  │ Engine    │                    │ Features:  20–50ms  │     │
│  └────┬─────┘                    │ Inference: 5–10ms   │     │
│       │                          │ Overhead:  20–40ms  │     │
│       ▼                          │ ─────────────────── │     │
│  ┌──────────────┐                │ Total:   ~100–200ms │     │
│  │ Feature       │               └─────────────────────┘     │
│  │ Service       │  ~20–50 ms                                │
│  │               │                                           │
│  │ ┌───────────────────────────────────────┐                 │
│  │ │ Static features  → in-memory cache    │                 │
│  │ │ Historical aggs  → precomputed hourly │                 │
│  │ │ Real-time traffic → Redis (TTL ~2min) │                 │
│  │ │ Weather          → cached (TTL ~15min)│                 │
│  │ │ Incidents        → Redis (TTL ~5min)  │                 │
│  │ └───────────────────────────────────────┘                 │
│  └────┬─────────┘                                            │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │ Model     │  ~5–10 ms                                     │
│  │ Server    │  Stage 1 (segment) → Stage 2 (route)          │
│  │           │  LightGBM served via custom C++ or             │
│  │           │  Triton / ONNX Runtime                        │
│  └────┬─────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  ETA Response                                                │
│                                                              │
│  ┌─────────────────────────────────────────┐                 │
│  │  LIVE RE-ESTIMATION (during navigation)  │                │
│  │                                          │                │
│  │  Every ~30s while user is driving:       │                │
│  │  - Subtract completed segments           │                │
│  │  - Re-query traffic for remaining route  │                │
│  │  - Re-run model on remaining segments    │                │
│  │  - Smooth update (avoid jittery ETA)     │                │
│  │    → exponential moving avg on ETA       │                │
│  └─────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
```

## 10. Evaluation & Deployment

### Offline Evaluation
- MAPE by trip distance bucket (short <5km, medium 5–30km, long >30km)
- MAPE by time-of-day (rush hour vs off-peak)
- MAPE by road type mix (highway-heavy vs urban)
- Underestimate rate (% trips arriving >10% later than predicted)

### Online A/B Testing
- Control: current production model
- Treatment: new model
- Metric: % trips within ±10% ETA, user re-engagement
- Duration: 1–2 weeks, segment by region

### Monitoring & Alerts
- Real-time MAPE dashboard (rolling 1hr window)
- Alert if MAPE drifts >X% from baseline
- Alert on feature drift (e.g., traffic feed goes stale)
- Segment-level error heatmap (catch regional degradation)

### Failure Modes & Mitigations
| Failure | Impact | Mitigation |
|---------|--------|------------|
| Traffic feed goes down | Stale speed data → bad ETA | Fall back to historical avg speeds |
| New road not in graph | Route can't use it | Regular graph updates (weekly) |
| Major event (concert, disaster) | Unprecedented congestion | Incident feature + manual override |
| Model latency spike | >300ms response | Timeout → fall back to baseline |
| GPS trace quality drops | Bad training labels | Filter outlier trips (speed > 200km/h, etc.) |

## 11. Key Design Decisions — Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | LightGBM (two-stage) | Fast, strong on tabular, interpretable |
| Architecture | Segment + route residual | Granular + captures cross-segment effects |
| Metric | MAPE + asymmetric penalty | Relative error + penalize underestimates |
| Retraining | Weekly | Traffic patterns shift seasonally |
| Fallback | Baseline segment summation | Always available, no ML dependency |
| Live updates | Re-estimate every ~30s | Keep ETA accurate during navigation |

## 12. System-Level Considerations

### Automatic Feedback Loop
```
User navigates trip
       │
       ▼
Trip completes → actual_time recorded
       │
       ▼
(predicted_ETA, actual_time) → training pipeline
       │
       ▼
Model improves continuously — no human labeling needed
```
- Every completed trip = free labeled example
- Enables continuous model improvement
- Detect degradation early: if recent prediction errors spike, trigger alert

### Routing ↔ ETA Coupling (Causal Feedback Loop)
```
ETA model says Route A is faster
        │
        ▼
Routing engine sends users to Route A
        │
        ▼
Route A gets congested
        │
        ▼
ETA prediction was wrong ← !!!
```
- **Risk:** oscillation — model drives traffic, which invalidates model
- **Mitigation 1:** real-time traffic signal captures current congestion
  regardless of cause — dampens the loop
- **Mitigation 2:** avoid aggressive rerouting (only reroute if
  delta > threshold, e.g., >5 min savings)
- **Mitigation 3:** use exploration — occasionally show suboptimal
  routes to collect unbiased data (like ε-greedy)

### Cold Start — Sparse Data Segments
- New roads or low-traffic areas have few GPS traces
- Fall back to: road-type average speed (e.g., avg speed for all
  "residential" roads in the same city)
- As data accumulates, segment-specific estimates take over

## 13. Future Improvements
1. **Graph Neural Network** — model road network topology directly (Google DeepETA)
2. **Supersegments** — group consecutive segments to reduce inference calls
3. **Personalization** — adjust for individual driving style (aggressive vs cautious)
4. **Confidence intervals** — show "15–20 min" instead of "17 min"
5. **Multi-route ranking** — use ETA model to rank alternative routes