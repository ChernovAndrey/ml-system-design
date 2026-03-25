# Bot Detection System — Social Network (10M DAU)

## Requirements
- 10M DAU, batch detection (daily)
- Manual review capacity: ~1000 accounts/day
- False negatives (missed bots) are worse than false positives
- Flagged users → manual review queue

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                         │
│  Profile metadata | Behavior logs | Social graph | UGC  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING (Daily Batch)           │
│  Account-level | Behavioral | Graph | Content           │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   SCORING PIPELINE                      │
│                                                         │
│  Layer 1: Rule-based filters  (cheap, high-precision)   │
│        ▼                                                │
│  Layer 2: ML model (GBT — XGBoost/LightGBM)            │
│        ▼                                                │
│  Layer 3: Threshold / Tiering                           │
│    ┌──────────┬──────────────┬──────────────┐           │
│    │ HIGH     │ MEDIUM       │ LOW          │           │
│    │ score≥0.95│ 0.7≤score<0.95│ score<0.7  │           │
│    │ Auto-ban │ → Review Q   │ Monitor      │           │
│    └──────────┴──────────────┴──────────────┘           │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              MANUAL REVIEW QUEUE                        │
│  ~950 top-scored + ~50 random sample (bias correction)  │
│  Reviewer sees: profile, activity, top features         │
│  Decision: Bot / Not Bot / Needs More Info              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              ACTIONS & FEEDBACK LOOP                    │
│  Bot → ban/suspend    |  Not bot → release              │
│  All decisions → label store → retraining pipeline      │
└─────────────────────────────────────────────────────────┘
```

---

## 1. Baseline: Rule-Based System (v0)

| Rule                                    | Signal                       |
|-----------------------------------------|------------------------------|
| Account age < 1d + >50 posts            | Burst posting, new account   |
| Default avatar + no bio + follows > 100 | Incomplete profile + mass follow |
| Identical messages to 10+ users         | Spam pattern                 |
| Login from known datacenter IP          | Infra signal                 |
| Posting frequency > 5σ above mean       | Superhuman activity rate     |

---

## 2. Features (for ML model)

### Account features
- Account age (days)
- Profile completeness score (avatar, bio, link — 0 to 1)
- Username entropy (random strings → suspicious)
- Email domain (free vs. corporate vs. disposable)
- Verification status

### Behavioral features
- Posts/day, likes/day, follows/day
- Avg time between consecutive actions (bots = low variance)
- **Action regularity score** — std dev of inter-action intervals
  (humans are noisy; bots are metronome-like)
- Session count & avg session length
- Time-of-day distribution entropy (humans sleep; bots don't)
- Action type diversity (post/like/follow/comment ratio)

### Social graph features
- Follower / following ratio
- Reciprocity rate (what % of follows are mutual)
- Clustering coefficient (do your friends know each other?)
- **Bot-neighbor score** — % of connections already flagged as bots
  (bots cluster — this is one of the strongest signals)
- Avg account age of followers

### Content features
- Self-similarity: avg pairwise cosine sim of user's posts
  (high = repetitive/templated)
- Link/URL ratio in posts
- Hashtag-to-word ratio
- Avg post length variance

---

## 3. Training Data & Labeling

### Cold-start (before model exists)
| Source                    | Quality    | Volume      |
|---------------------------|-----------|-------------|
| Rule-based flag labels    | Weak +    | High        |
| Manual seed labeling      | Gold      | ~2-5K       |
| Behavioral honeypots      | Strong +  | Low-Medium  |
| Historical ban data       | Gold      | Varies      |

### Steady-state (after model deployed)
- Primary: reviewer decisions from the queue (~1000/day)
- **Selection bias fix:** reserve ~50 slots/day for random
  low-scored accounts → calibrates FN rate
- Retrain model weekly on rolling 90-day label window

### Label quality
- Inter-annotator agreement tracking (double-review 10% of queue)
- Reviewer guidelines with clear examples of bot vs. not-bot
- "Needs More Info" option to avoid forced low-quality labels

---

## 3b. Handling Class Imbalance

**Expected imbalance:**
- Raw population: ~2-5% bots → 20:1 to 50:1 ratio
- Training set (enriched via review queue): ~10-20% bots → 5:1 to 10:1
  (because we oversample suspicious accounts by design)

**Strategy: combine multiple approaches**

```
 ┌──────────────────────────────────────────────────────┐
 │           IMBALANCE HANDLING STRATEGY                │
 │                                                      │
 │  1. Training time: scale_pos_weight in LightGBM     │
 │     (= ratio neg/pos, e.g. 8.0)                     │
 │     → model penalizes FN more than FP               │
 │                                                      │
 │  2. Data: stratified k-fold cross-validation         │
 │     → every fold preserves class ratio               │
 │                                                      │
 │  3. Post-training: threshold calibration             │
 │     → tune threshold on validation set to optimize   │
 │       Precision@1000 (our actual business objective) │
 │                                                      │
 │  4. Evaluation: PR-AUC, not ROC-AUC or accuracy     │
 │     → accuracy is meaningless at 95%+ majority class │
 └──────────────────────────────────────────────────────┘
```

| Approach                  | Use?  | Why / Why not                              |
|---------------------------|-------|--------------------------------------------|
| scale_pos_weight          | YES   | Simple, native to LightGBM, no data loss   |
| Stratified CV             | YES   | Ensures stable evaluation across folds      |
| Threshold calibration     | YES   | Most impactful lever for deployment         |
| PR-AUC as primary metric  | YES   | Robust under imbalance (unlike ROC/accuracy)|
| Undersampling majority    | NO    | Loses normal-behavior diversity → more FP   |
| SMOTE / oversampling      | NO    | Synthetic bots ≠ real bots; artifacts risk  |
| Focal loss                | NO    | For NNs, not applicable to GBTs             |

**Why threshold calibration matters most:**
- Model outputs raw scores, not calibrated probabilities
- Default 0.5 cutoff is almost never optimal under imbalance
- We tune threshold to maximize recall *subject to* the review
  queue fitting within 1000/day capacity
- Separate calibration step on held-out validation set

---

## 4. Model Choice: Gradient Boosted Trees

| Option           | Pros                              | Cons                          |
|------------------|-----------------------------------|-------------------------------|
| Logistic Reg.    | Simple, interpretable, fast       | Misses non-linear interactions|
| **GBT (LightGBM)** | **Strong on tabular, feature importances, handles missing values** | **Moderate complexity** |
| Neural Network   | Can ingest raw sequences          | Overkill for tabular, less interpretable |
| GNN (graph)      | Captures network topology         | Complex infra, slow training  |

**Decision:** Start with LR as v0 baseline → graduate to LightGBM as v1.
Consider GNN as a future enhancement for graph-heavy signals.

**Why GBT wins here:**
- Tabular features → GBTs dominate (well-established in literature)
- Feature importances → reviewers see *why* an account was flagged
- Handles mixed feature types and missing values natively
- Fast: scores 10M accounts in minutes on modest hardware
- Easy to iterate: add/remove features without architecture changes

---

## 5. Review Queue Design

```
Daily pipeline output (~10M scores)
         │
         ├─ score ≥ 0.95 ──→ AUTO-ACTION (ban/restrict)
         │                    ~N accounts/day (monitor volume)
         │
         ├─ 0.7 ≤ score < 0.95 ──→ REVIEW QUEUE
         │                          sorted by score DESC
         │                          serve top ~950 to reviewers
         │
         ├─ score < 0.7 ──→ MONITOR (re-score tomorrow)
         │
         └─ random sample from score < 0.7 ──→ 50/day to queue
                                                (bias correction)
```

**Reviewer experience:**
- Account profile snapshot
- Recent activity timeline
- Top 5 contributing features with values
- Network visualization (who do they follow/interact with)
- One-click decision: Bot / Legitimate / Unsure

---

## 6. Evaluation & Metrics

### Offline metrics
- **Precision@1000** — of the 1000 sent to review, how many are bots?
  (directly maps to reviewer efficiency)
- **Recall** — what fraction of all bots do we catch?
- **PR-AUC** — precision-recall tradeoff across thresholds
  (prefer over ROC-AUC because classes are imbalanced)

### Online metrics
- **Bot prevalence** — estimated % of active bots on platform
  (measured via random audits)
- **Reviewer throughput** — decisions/day, avg time per decision
- **Appeal overturn rate** — how often do banned users
  successfully appeal? (proxy for FP in auto-ban tier)
- **User reports of bot activity** — leading indicator of FN

### Guardrail metrics
- Auto-ban volume/day (alert if spike → model may be miscalibrated)
- Queue overflow rate (are we generating > 1000 medium-conf/day?)

---

## 7. Monitoring & Iteration

### Model monitoring
- **Feature drift** — track distribution of top features daily
  (if bots change behavior, features shift)
- **Score distribution drift** — if avg score shifts significantly,
  investigate (new bot wave? or model decay?)
- **Label-prediction agreement** — reviewer decisions vs. model
  scores over time (should stay correlated)

### Adversarial adaptation
Bots WILL evolve. Mitigation:
- **Retrain weekly** on fresh labels
- **Feature refresh** — quarterly review of feature importance;
  retire gamed features, add new ones
- **Ensemble diversity** — maintain rule-based + ML layers so
  gaming one doesn't defeat both
- **Honeypot rotation** — change bait content/traps regularly

### Failure modes & mitigations
| Failure                    | Impact            | Mitigation                     |
|---------------------------|-------------------|--------------------------------|
| Model scores spike (false alarm) | Mass false bans | Auto-ban volume circuit breaker |
| Sophisticated bots score < 0.7  | Missed bots     | Random sampling catches drift   |
| Review queue overwhelmed  | Backlog grows     | Dynamic threshold adjustment    |
| Training data poisoned    | Model corrupted   | Reviewer agreement checks, anomaly detection on labels |
| Feature pipeline failure  | Stale scores      | Pipeline health alerts, fallback to rules |

---

## 8. Deployment & Iteration Roadmap

| Phase   | What                                      | Timeline  |
|---------|-------------------------------------------|-----------|
| v0      | Rule-based filters only                   | Week 1-2  |
| v1      | Logistic regression + rules               | Week 3-4  |
| v2      | LightGBM + rules + review queue           | Week 5-8  |
| v3      | Add graph features, tune thresholds       | Month 3   |
| Future  | GNN for network analysis, NLP on content  | Month 6+  |

---

## Key Design Decisions Summary

| Decision                  | Choice            | Reason                                          |
|---------------------------|-------------------|-------------------------------------------------|
| Batch vs. real-time       | Batch (daily)     | Acceptable latency; enables richer features     |
| Model                     | LightGBM          | Best for tabular data, interpretable            |
| Auto-ban threshold        | 0.95              | FN > FP, but auto-ban must be high-confidence   |
| Review queue size         | 950 + 50 random   | Max capacity with bias correction               |
| Retraining cadence        | Weekly             | Balance freshness vs. stability                 |