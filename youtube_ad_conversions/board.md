# YouTube Ad Conversion Prediction System

## 1. Problem Statement
- Predict P(conversion | user, ad, context) in real-time
- Conversion = any positive ad interaction (website visit, app install, sign-up, purchase)
- Used in ad auction: expected_value = bid × P(click) × P(conversion | click)
- Scale: billions of impressions/day, thousands of campaigns

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ONLINE SERVING                         │
│                                                             │
│  Ad Request ──► Feature Assembly ──► Model Inference ──► Score
│  (user_id,       (user features,      (pCVR model)     (P_cvr)
│   context)        ad features,                              │
│                   cross features)                           │
│                       ▲                                     │
│                       │                                     │
│              ┌────────┴────────┐                            │
│              │  Feature Store  │                            │
│              │  (Redis/online) │                            │
│              └────────┬────────┘                            │
└───────────────────────┼─────────────────────────────────────┘
                        │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                        │
┌───────────────────────┼─────────────────────────────────────┐
│                 OFFLINE / NEARLINE                          │
│                       │                                     │
│  ┌────────────────────┴───────────────────┐                 │
│  │         Feature Pipeline               │                 │
│  │  (Spark / Flink — batch + streaming)   │                 │
│  └──────────┬─────────────────┬───────────┘                 │
│             │                 │                              │
│   ┌─────────▼──────┐  ┌──────▼────────┐                    │
│   │  User Behavior │  │  Ad/Campaign  │                    │
│   │  Logs (clicks, │  │  Metadata     │                    │
│   │  watches, etc) │  │               │                    │
│   └────────────────┘  └───────────────┘                    │
│                                                             │
│  ┌──────────────────────────────────────────┐               │
│  │         Training Pipeline                │               │
│  │  Label Join ──► Training ──► Validation  │               │
│  │  (attribution    (model)     (offline     │               │
│  │   window)                     eval)       │               │
│  └──────────────────────────────────────────┘               │
│                                                             │
│  ┌──────────────────────────────────────────┐               │
│  │         Conversion Attribution           │               │
│  │  Impression Log + Conversion Pixel/API   │               │
│  │  ──► Join on user_id within time window  │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data & Features

### Label Definition
- Positive: user converts within attribution window (e.g., 7 days) after ad impression
- Negative: user does NOT convert within window
- Challenge: delayed labels → need to wait before labeling negatives

### Feature Groups

| Group | Examples | Storage |
|-------|----------|---------|
| **User features** | demographics, watch history, past ad interactions, device, geo | Feature Store (batch) |
| **Ad/Campaign features** | ad creative type, vertical, landing page, campaign age, historical CVR | Feature Store (batch) |
| **Context features** | time of day, day of week, video category being watched, device | Real-time from request |
| **Cross features** | user×ad_category affinity, user×advertiser history | Computed at serving or pre-computed |
| **Real-time features** | # ads seen in session, recency of last conversion, session length | Streaming pipeline |

---

## 4. Model Approach

### Baseline
- Logistic Regression on campaign-level historical CVR + user segment features
- Simple, interpretable, fast — good starting point

### Production Model — Wide & Deep

```
                    ┌──────────────┐
                    │   P(convert) │
                    │   sigmoid    │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │   Combine    │
                    │  (concat +   │
                    │   dense)     │
                    └──┬───────┬───┘
                       │       │
          ┌────────────┘       └────────────┐
          │                                 │
   ┌──────┴───────┐                ┌────────┴────────┐
   │  WIDE SIDE   │                │   DEEP SIDE     │
   │              │                │                 │
   │ Hand-crafted │                │ Dense layers    │
   │ feature      │                │ (256→128→64)    │
   │ crosses:     │                │       ▲         │
   │              │                │       │         │
   │ • user_seg   │                │ ┌─────┴──────┐  │
   │   × ad_vert  │                │ │ Embedding  │  │
   │ • user_hist  │                │ │ Layer      │  │
   │   × advertis │                │ │            │  │
   │ • device     │                │ │ user_id    │  │
   │   × ad_type  │                │ │ ad_id      │  │
   │              │                │ │ campaign   │  │
   │              │                │ │ video_cat  │  │
   │              │                │ │ advertiser │  │
   └──────────────┘                └─┴────────────┴──┘
```

- **Wide side**: memorizes specific high-signal combinations
  (e.g., "users in segment 'tech enthusiasts' × ad vertical 'electronics' → high CVR")
- **Deep side**: learns generalizable representations through embeddings
  (e.g., can infer that a user who watches cooking videos might convert on kitchen gadget ads
   even if that exact cross was never observed)
- Trained jointly, end-to-end with log-loss

### Why Wide & Deep over alternatives?
- **vs. plain LR**: LR can't generalize to unseen feature crosses
- **vs. pure DNN**: loses memorization of specific high-value crosses
- **vs. Two-Tower**: Two-tower is great for retrieval (candidate generation),
  but we're in the ranking/scoring phase — we WANT feature interaction, not dot-product similarity
- Trade-off: more complex to train/serve than LR, but the lift in conversion
  prediction accuracy directly translates to auction revenue

### Training Strategy

**Delayed Feedback Problem (critical)**
```
Timeline:
  t=0          t=1h         t=1d          t=7d
  ├── ad shown ──┼── click ───┼── ??? ──────┼── purchase!
                                              └─ label = 1
  Problem: at t=1d, this looks like a negative, but it's actually positive
```
- Approach 1: **Wait-then-train** — only use examples older than 7 days
  → simple but 7-day model freshness lag
- Approach 2: **Importance weighting** — use fresh examples but upweight
  based on P(conversion still coming | elapsed time)
  → better freshness, more complex
- I'd start with Approach 1 for simplicity, move to Approach 2 if freshness matters

**Class Imbalance** (~0.1–1% CVR)
- Negative downsampling: keep all positives, sample ~10% of negatives
- Then apply calibration correction: p_calibrated = p_raw / (p_raw + (1-p_raw)/sampling_rate)
- Why not just focal loss? Downsampling also reduces training cost significantly

**Training Cadence**
- Daily retraining on rolling 30-day window
- Why daily, not hourly? Conversion labels need time to materialize.
  Retraining too frequently means training on incomplete labels.

---

## 5. Evaluation & Metrics

### Offline
- **AUC-ROC** — ranking quality
- **Log-loss / calibration** — critical since scores feed into auction math
- **PR-AUC** — better for imbalanced data
- Calibration plot: predicted vs actual conversion rates by decile

### Online
- **A/B test metrics:**
  - Conversion rate lift
  - Revenue per 1000 impressions (RPM)
  - Advertiser ROI / ROAS
  - User experience metrics (ad fatigue, watch time impact)

---

## 6. Serving Architecture & Latency

### Where pCVR sits in the ad auction pipeline
```
User opens video
      │
      ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Candidate   │     │   Predict    │     │    Rank &    │     │  Serve   │
│  Generation  │────►│  pCTR, pCVR  │────►│   Auction    │────►│  Winner  │
│              │     │  per (user,  │     │              │     │          │
│  ~1000 ads   │     │   ad) pair   │     │ score = bid  │     │  1 ad    │
│  eligible    │     │              │     │  × pCTR      │     │          │
│              │     │  ~100 ads    │     │  × pCVR      │     │          │
│  (targeting  │     │  (pre-filter │     │              │     │          │
│   match)     │     │   by pCTR)   │     │  2nd price   │     │          │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────┘
     ~5ms                ~5-10ms               ~2ms               ~1ms
                      ▲ OUR MODEL
```
- Total auction budget: ~20-30ms
- Our pCVR model gets ~5-10ms (inference + feature lookup)
- We only score ~100 candidates (after pCTR pre-filtering), not all 1000

### Serving Flow (detailed)
```
Ad Request (user_id, video_id, device, geo, timestamp)
      │
      ├──────────────────────────────────┐
      │                                  │
      ▼                                  ▼
┌─────────────┐                  ┌───────────────┐
│ Feature     │                  │ Context       │
│ Store       │                  │ Features      │
│ (Redis)     │                  │ (from request)│
│             │                  │               │
│ • user emb  │                  │ • time of day │
│ • user hist │                  │ • device      │
│ • ad feats  │                  │ • video cat   │
│ • camp CVR  │                  │ • session len │
└──────┬──────┘                  └───────┬───────┘
       │          ┌──────────┐           │
       └─────────►│ Feature  │◄──────────┘
                  │ Assembly │
                  │ (concat, │
                  │  crosses)│
                  └────┬─────┘
                       │
                       ▼
              ┌────────────────┐
              │  Model Server  │
              │  (TF Serving / │
              │   Triton)      │
              │                │
              │  Batched infer │
              │  ~100 ads at   │
              │  once per user │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │  Calibration   │
              │  Layer         │
              │  (correct for  │
              │  downsampling) │
              └────────┬───────┘
                       │
                       ▼
                 pCVR scores
                 → auction
```

### Latency Optimizations
1. **Batch scoring**: score all ~100 candidate ads in ONE model call (batch inference)
   - GPU-friendly, amortizes overhead
2. **Parallel feature fetch**: user features + ad features fetched concurrently
3. **Pre-computed embeddings**: user embeddings updated hourly via batch pipeline,
   NOT computed at serving time
4. **Feature store locality**: Redis cluster co-located with serving, <1ms lookup
5. **Model quantization**: INT8 quantization of the deep network → 2-3x speedup
   with minimal accuracy loss (<0.1% AUC drop typically)

### Fallback & Resilience
- Feature store timeout (>3ms): use default/mean feature vectors → graceful degradation
- Model server down: fall back to simpler LR model (always warm) or campaign-level CVR prior
- Circuit breaker pattern: if error rate >5%, auto-switch to fallback for 30s

---

## 7. Monitoring & Failure Modes

### Real-time Monitoring
```
┌───────────────────────────────────────────────────────┐
│                    Monitoring Dashboard                │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ Model       │  │ System      │  │ Business     │  │
│  │ Health      │  │ Health      │  │ Health       │  │
│  │             │  │             │  │              │  │
│  │ • pred dist │  │ • p99 lat   │  │ • conv rate  │  │
│  │   shift     │  │ • error %   │  │ • RPM        │  │
│  │ • calib     │  │ • feat miss │  │ • advertiser │  │
│  │   error     │  │   rate      │  │   ROAS       │  │
│  │ • AUC trend │  │ • throughput│  │ • fill rate  │  │
│  └─────────────┘  └─────────────┘  └──────────────┘  │
│                                                       │
│  Alert Rules:                                         │
│  • Calibration error > 5% → page oncall              │
│  • p99 latency > 15ms → auto-scale + alert           │
│  • Prediction mean shifts >20% in 1h → block deploy  │
└───────────────────────────────────────────────────────┘
```

### Key Failure Modes
| Failure | Impact | Mitigation |
|---------|--------|------------|
| **Calibration drift** | Auction math breaks, wrong ads win | Daily recalibration, isotonic regression as safety net |
| **Stale features** | User embeddings outdated | TTL on feature store entries, alert if refresh lag >2h |
| **Label pipeline delay** | Training on incomplete labels | Monitor label arrival curves, pause retraining if anomalous |
| **Cold start (new campaign)** | No historical CVR | Hierarchical fallback: campaign→advertiser→vertical→global prior |
| **Adversarial advertisers** | Gaming conversion signals | Conversion validation, anomaly detection on conversion patterns |
| **Seasonal shifts** | Holiday traffic ≠ normal traffic | Time-based features + more frequent retraining during peak seasons |

---

## 8. Deployment & Iteration

### Safe Rollout
1. Shadow mode: new model scores alongside production, no traffic impact
2. Compare offline metrics (AUC, calibration) old vs new
3. 1% A/B test → measure conversion rate, RPM, advertiser ROAS
4. Ramp: 1% → 5% → 25% → 50% → 100% over ~1 week
5. Auto-rollback if key metrics degrade >2% for >1 hour

### Iteration Roadmap
- **v1**: LR baseline with campaign CVR + user segments
- **v2**: Wide & Deep with embeddings
- **v3**: Add real-time session features (streaming pipeline)
- **v4**: Value-weighted optimization (predict conversion VALUE, not just binary)
- **v5**: Multi-task learning (pCTR + pCVR jointly, shared representations)

---

## 8. Key Design Decisions (Trade-offs)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Attribution window | 7 days | Balances coverage vs label delay |
| Negative sampling | Downsample + calibrate | Needed at YouTube scale |
| Model architecture | Wide & Deep | Balance memorization + generalization |
| Retraining cadence | Daily | Captures trend shifts without instability |
| Cold start | Segment priors | Avoids zero predictions for new entities |