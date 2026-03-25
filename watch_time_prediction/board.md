# ML System Design — Netflix Watch Time Prediction

## Problem Statement
Predict % of content a user will watch for a given movie/series (0–100%).
- **When:** Pre-click, at ranking time
- **Latency:** ~10-50ms for hundreds of candidates
- **Purpose:** Drive better recommendations by ranking on predicted engagement depth

---

## 1. Metrics

### Offline Metrics
- **Primary:** MAE / RMSE on predicted vs. actual % watched
- **Secondary:** NDCG@K — does the model rank high-completion titles above low-completion ones?
- **Bucketed accuracy:** Prediction quality across bins (0-25%, 25-50%, 50-75%, 75-100%)

### Online Metrics (A/B test)
- **Primary:** Average completion rate across all views (did recs improve engagement?)
- **Guardrails:**
  - Total hours watched (don't reduce volume)
  - Content diversity (don't collapse into filter bubble)
  - Retention / churn rate
  - Click-through rate (don't tank discovery)

---

## 2. Baseline

**Simple baseline:** Weighted average of:
- Global avg completion rate for that title
- User's historical avg completion rate
- Genre-level avg completion for user × genre pair

Prediction = α · title_avg + β · user_avg + γ · user_genre_avg

Fast, interpretable, gives a floor to beat.

---

## 3. Features

### User Features
- Demographics (age, country, language, subscription tier)
- Viewing history stats (avg completion rate, avg session length, #titles watched)
- Genre/taste profile (distribution of genres watched, top genres by completion)
- Temporal patterns (time-of-day, day-of-week preferences, binge tendency)
- Recency-weighted engagement (recent completion rates matter more)
- Device preferences (TV vs. mobile vs. tablet)

### Content (Item) Features
- Metadata: genre, runtime, release year, rating (TV-MA, PG-13), #seasons
- Popularity: global avg completion, click-through rate, trending score
- Content quality signals: awards, critic scores, cast/director popularity
- Content embeddings (from poster, synopsis, trailer — pre-computed)
- Freshness: days since release, days since added to Netflix

### Context Features
- Time of day, day of week (current session)
- Device type
- How far into a browsing session the user is
- What the user recently watched (last 1-3 titles and their completion %)

### Cross Features (User × Item interactions)
- User's completion rate for this genre
- User's completion rate for similar titles (collaborative filtering signal)
- User's affinity to cast/director (watched other titles by same people?)
- User watched trailer / added to My List? (strong intent signals)

---

## 4. Model Architecture

### Training Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                   OFFLINE TRAINING                        │
│                                                          │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ Viewing  │──▶│   Feature    │──▶│   Training Data  │  │
│  │  Logs    │   │  Engineering │   │  (user, item,    │  │
│  │ (Spark)  │   │   Pipeline   │   │   context, label)│  │
│  └─────────┘   └──────────────┘   └────────┬─────────┘  │
│                                             │            │
│                                             ▼            │
│                                    ┌────────────────┐    │
│                                    │  Model Training │    │
│                                    │  (see below)    │    │
│                                    └────────┬───────┘    │
│                                             │            │
│                                             ▼            │
│                                    ┌────────────────┐    │
│                                    │  Model Registry │    │
│                                    └────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### Model Choice: Two-stage approach

**Stage 1 — Gradient Boosted Trees (GBDT, e.g., XGBoost/LightGBM)**
- Strong baseline model
- Handles tabular features well
- Fast inference, easy to debug
- Naturally handles missing values, feature interactions

**Stage 2 — Deep Learning model (if GBDT plateau is reached)**

```
User Features ──▶ ┌──────────┐
                   │  User    │
                   │  Tower   │──┐
                   └──────────┘  │
                                 │   ┌────────────┐    ┌──────────┐
                                 ├──▶│   Cross    │───▶│  Output  │
                                 │   │  Network + │    │  Head    │
Item Features ──▶ ┌──────────┐  │   │    DNN     │    │ (sigmoid │
                   │  Item    │──┤   └────────────┘    │  × 100)  │
                   │  Tower   │  │                     └──────────┘
                   └──────────┘  │
                                 │
Context Feats ──▶ ┌──────────┐  │
                   │ Context  │──┘
                   │  Layer   │
                   └──────────┘
```

- Two-tower style with cross-network (DCN-v2 inspired)
- User tower: encodes user profile + viewing history (can include sequence model)
- Item tower: encodes content features + embeddings
- Cross network captures explicit feature interactions
- Output: sigmoid activation scaled to [0, 100] for % prediction
- Loss: MSE or Huber loss (robust to outliers)

**Why this over a single tower?**
- Item tower embeddings can be precomputed → faster serving
- User tower can be partially cached and updated incrementally

### Handling Series vs. Movies
- For **movies**: predict % of single movie watched
- For **series**: predict % of next episode watched (episode-level)
  - Additional features: episodes already watched, binge velocity,
    time since last episode, season progression

---

## 5. Training Data & Label Definition

### Label: % watched
- label = min(time_watched / total_runtime, 1.0) × 100
- **Filter out:** accidental clicks (< 30 seconds or < 2%)
- **Handle rewatches:** use first-watch only, or flag rewatch as feature
- **Negative sampling:** user was shown title but didn't click → label is 0%?
  - Trade-off: introduces noise, but captures "would not watch" signal
  - Alternative: only train on clicked titles, adjust for selection bias

### Training Window
- 6-12 months of viewing data
- Recent data weighted higher (user tastes drift)
- ~100M+ training examples at Netflix scale

---

## 6. Serving Architecture

```
┌──────────┐    ┌───────────────┐    ┌─────────────────┐
│  User    │───▶│   Candidate   │───▶│  Feature Store  │
│ Request  │    │  Generation   │    │  (Redis/online) │
└──────────┘    │  (~1000 items)│    └────────┬────────┘
                └───────────────┘             │
                                              ▼
                 ┌─────────────────────────────────────┐
                 │        Watch Time Predictor          │
                 │                                      │
                 │  ┌────────────┐  ┌───────────────┐  │
                 │  │ User Emb   │  │ Item Emb      │  │
                 │  │ (cached)   │  │ (precomputed) │  │
                 │  └─────┬──────┘  └──────┬────────┘  │
                 │        │                │            │
                 │        ▼                ▼            │
                 │  ┌──────────────────────────┐       │
                 │  │  Scoring Layer           │       │
                 │  │  (cross net + DNN head)  │       │
                 │  └────────────┬─────────────┘       │
                 │               │                      │
                 └───────────────┼──────────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  Ranking + Business Rules     │
                 │  (diversity, freshness boost, │
                 │   new content exploration)    │
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  Final Ranked Feed → User UI  │
                 └───────────────────────────────┘
```

### Latency Optimization
- Item embeddings precomputed offline → stored in feature store
- User embeddings computed on login / periodically refreshed
- Scoring layer is lightweight dot-product + small DNN → <10ms
- Batch scoring: score all candidates in one GPU/vectorized call
- Model distillation: train smaller model for serving if needed

---

## 7. Evaluation & Deployment

### Offline Evaluation
- Train/val/test split by TIME (not random) — prevent future leakage
- Compare: baseline → GBDT → deep model
- Slice analysis: new users, new content, genre, device, country

### Online A/B Test
- Control: current ranking system
- Treatment: ranking incorporating watch-time prediction
- Measure: avg completion rate, total hours, retention, diversity
- Run for 2-4 weeks, check for novelty effects

### Monitoring
- Prediction distribution drift (model predictions shifting over time)
- Feature drift (input distributions changing)
- Completion rate by prediction bucket (calibration)
- Latency p50/p95/p99
- Alert on: sudden prediction mean shift, feature pipeline delays

### Iteration Plan
- V1: GBDT with tabular features → prove value
- V2: Add content embeddings (text, poster)
- V3: Deep model with sequence modeling (user watch history as sequence)
- V4: Multi-task learning (predict completion % + click probability jointly)

---

## 8. Key Design Decisions & Trade-offs

| Decision | Choice | Why | Revisit if... |
|----------|--------|-----|----------------|
| Prediction target | % completion | Directly measures engagement depth | We need session-level optimization |
| Model | GBDT first, then DNN | Fast iteration, then incremental gains | GBDT plateaus quickly |
| Negative samples | Train on clicks only + debias | Cleaner labels | We have good selection bias correction |
| Series handling | Episode-level prediction | Granular, captures drop-off | Too complex → aggregate to season |
| Architecture | Two-tower + cross | Precomputable embeddings, fast serving | Latency is not a concern → single tower |
| Loss function | Huber loss | Robust to outliers (partial views, falls asleep) | Distribution is clean → switch to MSE |

---

## 9. Failure Modes & Mitigations

- **Cold-start users:** Fall back to popularity + demographic-based prediction
- **Cold-start content:** Use content features only (genre, cast, synopsis embedding)
- **Popularity bias:** Diversity re-ranking layer ensures exploration
- **Stale predictions:** User embeddings refreshed on each session; model retrained weekly
- **Prediction outliers:** Clip predictions to [0, 100], monitor calibration

---

## 10. Selection Bias & Feedback Loops (Critical)

This is probably the most subtle and important issue in this system.

**The problem:** We only observe watch time for content the user *chose* to click.
Our training data is biased — we never see how long a user would watch a title
they were never shown or never clicked. If we naively train on this, the model
reinforces existing patterns rather than discovering new good matches.

**Mitigations:**
- **Inverse Propensity Weighting (IPW):** Weight training samples by 1/P(click),
  so rare clicks count more. Downweights "easy" predictions for popular content.
- **Position-debiased training:** Content in row 1, position 1 gets clicked more
  regardless of quality. Include position as feature during training, then set
  position=0 at inference to remove its influence.
- **Exploration traffic:** Reserve ~5% of impressions for random/epsilon-greedy
  exploration. This gives us unbiased data to evaluate the model and discover
  new content-user matches.

```
  Feedback Loop:

  Model predicts high ──▶ Title shown prominently ──▶ More clicks ──▶ More data
        ▲                                                                │
        └────────────── Model trains on this data ◀──────────────────────┘

  Without intervention, this loop narrows recommendations over time.
  Exploration traffic + IPW break the loop.
```

---

## 11. Multi-Task Learning (V4 detail)

Rather than predicting completion % in isolation, jointly predict:

```
                    ┌──────────────────┐
  Shared Layers ───▶│  Task 1: P(click)│  ← Binary cross-entropy
                    ├──────────────────┤
                    │  Task 2: % watch │  ← Huber loss
                    ├──────────────────┤
                    │  Task 3: P(finish)│  ← Binary (watched >90%)
                    └──────────────────┘
```

**Why multi-task?**
- Shared representation learns richer user-content understanding
- Click prediction helps with selection bias (models what gets clicked)
- P(finish) captures a high-value binary signal alongside the continuous %
- Final ranking score = f(P(click) × predicted_% × P(finish))
  This balances discovery (click) with depth (completion)

---

## 12. Personalization Depth — Handling User Segments

| User Segment       | Strategy                                         |
|--------------------|--------------------------------------------------|
| New user (0 views) | Content popularity + demographic prior            |
| Light user (1-10)  | Genre-level preferences + collaborative filtering |
| Active user (10+)  | Full personalized model with history sequence     |
| Returning (dormant)| Blend: 50% last-known prefs + 50% current trends |

Graceful degradation: model should work at every level, just with
increasing confidence as data grows.

---

## Summary of Key Insights

1. **Start simple** (GBDT) → iterate to deep models only when justified
2. **Selection bias is the #1 technical risk** — address with IPW + exploration
3. **Two-tower architecture** enables latency-efficient serving via precomputation
4. **Multi-task learning** aligns click + watch signals for better ranking
5. **Episode-level prediction** for series captures the binge/drop-off dynamic
6. **Feedback loops** must be actively broken with exploration traffic
7. **Graceful cold-start** degradation across user segments