# ML System Design — Instagram Feed Ranking

## Problem Statement
Given a user U and a set of candidate posts P, produce a **personalized ranked ordering**
of posts for the main home feed, optimizing for **engagement** (likes, comments, shares,
saves, dwell time).

**Constraints:** ~200ms latency budget for ranking. All raw data available.

---

## 1. High-Level Architecture

```
┌─────────────┐
│  User opens  │
│    feed      │
└─────┬───────┘
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│                CANDIDATE GENERATION                       │
│                                                           │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐ │
│  │ Source 1:     │ │ Source 2:     │ │ Source 3:         │ │
│  │ Follow-graph  │ │ Interest-    │ │ Trending /        │ │
│  │ (primary)     │ │ based ANN    │ │ Viral posts       │ │
│  └──────┬───────┘ └──────┬───────┘ └────────┬──────────┘ │
│         └────────────────┼──────────────────┘            │
│                          ▼                                │
│                  ┌──────────────┐                         │
│                  │   Merge +    │                         │
│                  │   Dedup +    │                         │
│                  │   Light      │                         │
│                  │   Pre-rank   │                         │
│                  └──────┬───────┘                         │
│                         │                                 │
└─────────────────────────┼────────────────────────────────┘
                          │  ~500-1000 candidates
      ▼
┌─────────────────┐     ┌───────────────────────┐
│  Ranking Model   │◄───│  Features (§3)         │
│  (multi-task     │     │  + Feature Store       │
│   scoring)       │     └───────────────────────┘
└─────┬───────────┘
      │  scored + sorted
      ▼
┌─────────────────┐
│  Post-Ranking    │
│  Rules           │
│  • diversity     │
│  • content-type  │
│    mixing        │
│  • policy/safety │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Final Feed      │
└─────────────────┘
```

---

## 1b. Candidate Generation — Deep Dive

### System trade-off: Fan-out on Write vs Read

| Approach         | How it works                          | Pro               | Con                        |
|------------------|---------------------------------------|--------------------|----------------------------|
| Fan-out on write | Push post to all follower inboxes     | Fast reads         | Expensive for celebrities  |
| Fan-out on read  | Pull from followed accounts at req    | No write amplif.   | Slower reads               |
| **Hybrid** ✓     | Write for <10K followers, read for rest| Bounded writes    | Extra complexity           |

### Source 1: Follow-graph retrieval (primary, ~80% of candidates)

Most of the feed is posts from accounts the user follows.

```
Alice follows 500 accounts
  → Fetch post IDs from pre-built inbox (fan-out-on-write)
  → For celebrity follows: query recent posts at read time
  → Filter: only posts since last session (or last 48h)
  → Result: ~2000 raw candidates
```

**Lightweight filtering** to get from ~2000 → ~800:
- Remove already-seen posts
- Remove policy-violating content
- Apply simple heuristic score: `0.3·author_affinity + 0.3·recency + 0.4·early_engagement`
- Keep top-K by this fast heuristic (no ML model, just weighted sum)

### Source 2: Interest-based retrieval (~15% of candidates)

For content from accounts the user does NOT follow — "suggested posts."

**Two-tower model for fast ANN retrieval:**
```
      TRAINING (end-to-end, joint optimization)
      ─────────────────────────────────────────

┌──────────────────────────────┐  ┌──────────────────────────────┐
│        User Tower             │  │        Post Tower              │
│                               │  │                                │
│  SPARSE features:             │  │  SPARSE features:              │
│  • user_id  → hash → embed(64)│  │  • author_id → hash → embed(64)│
│  • country  → embed(8)       │  │  • post_type → embed(8)        │
│  • age_bucket → embed(8)     │  │  • hashtags  → embed → pool    │
│                               │  │                                │
│  DENSE features (preprocessed)│  │  DENSE features (preprocessed) │
│  • interest_vec (128d)       │  │  • image_embed (frozen ViT,768d)│
│    (avg of engaged post       │  │  • caption_embed(frozen BERT)  │
│     embeds from pretrained    │  │  • early_engage: log-transform │
│     ViT — NOT from post tower)│  │    + bucketize → embed(8 each) │
│  • activity: log → bucketize │  │  • post_age: log → bucketize   │
│    → embed(8 each)           │  │    → embed(8)                  │
│    - sessions/day            │  │  • author_follower_count:       │
│    - posts_viewed/day        │  │    log → bucketize → embed(8)  │
│    - avg_session_duration    │  │                                │
│  • hist. eng rates:          │  │                                │
│    like_rate, comment_rate,  │  │                                │
│    share_rate, save_rate     │  │                                │
│    (already 0-1, use raw)    │  │                                │
│                               │  │                                │
│  concat all → ~400 dim       │  │  concat all → ~400 dim         │
│        │                      │  │        │                       │
│  ┌─────▼──────────┐          │  │  ┌─────▼──────────┐           │
│  │ MLP: 400→256   │ (ReLU)   │  │  │ MLP: 400→256   │ (ReLU)   │
│  │       256→128  │ (ReLU)   │  │  │       256→128  │ (ReLU)   │
│  └─────┬──────────┘          │  │  └─────┬──────────┘          │
│        │                      │  │        │                       │
│  ┌─────▼──────────┐          │  │  ┌─────▼──────────┐           │
│  │ Linear project │ (NO act.)│  │  │ Linear project │ (NO act.) │
│  │  128 → 128     │          │  │  │  128 → 128     │           │
│  └─────┬──────────┘          │  │  └─────┬──────────┘           │
│        │                      │  │        │                       │
│    L2 normalize               │  │    L2 normalize                │
│        │                      │  │        │                       │
│    user_vec (128d)            │  │    post_vec (128d)             │
└────────┬─────────────────────┘  └────────┬───────────────────────┘
          │                               │
          └───────── dot(u, p) ───────────┘
                       │
                    sigmoid
                       │
                  P(engagement)
                       │
              Binary Cross-Entropy Loss
              against label ∈ {0, 1}

  ⇒ Gradients flow back through BOTH towers jointly
  ⇒ Forces u and p into the SAME latent space
```

### Why does alignment happen?
- Both towers output to the SAME 128-dim space
- The ONLY way the model can minimize loss is by placing
  user vectors near posts they like, far from posts they don't
- L2 normalization → dot product = cosine similarity → bounded [-1, 1]
- Joint end-to-end training is what guarantees alignment

### Toy example of aligned embeddings:
```
After training, in the shared 128-dim space:

Alice (loves food + travel):     user_vec  = [0.8, 0.5, 0.1, 0.0, ...]
Bob   (loves fitness + fashion): user_vec  = [0.1, 0.0, 0.7, 0.6, ...]

Food post by @chef_mario:        post_vec  = [0.9, 0.3, 0.0, 0.1, ...]
Gym post by @fit_coach:          post_vec  = [0.0, 0.1, 0.8, 0.4, ...]

dot(Alice, food_post)  = 0.72 + 0.15 + 0.0 + 0.0 = 0.87  ✓ high
dot(Alice, gym_post)   = 0.0  + 0.05 + 0.08+ 0.0 = 0.13  ✗ low
dot(Bob,   food_post)  = 0.09 + 0.0  + 0.0 + 0.06= 0.15  ✗ low
dot(Bob,   gym_post)   = 0.0  + 0.01 + 0.56+ 0.24= 0.81  ✓ high
```

### ID Hashing — Handling Billion-Scale Categorical Features

**Problem:** 2B unique user IDs → full embedding table = ~512 GB. Won't fit.

**Solution:** Hash IDs into a fixed-size embedding table.

```
Full table (impossible):              Hashed table (feasible):
┌─────────────────────────┐           ┌────────────────────────┐
│ user_0001  → [0.2, ...]│           │ bucket_0 → [0.2, ...] │
│ user_0002  → [0.5, ...]│           │ bucket_1 → [0.5, ...] │
│ ...                     │           │ ...                    │
│ user_2B    → [0.1, ...]│           │ bucket_999999 → [...]  │
│                         │           │                        │
│ 2B rows × 64d = 512 GB │           │ 1M rows × 64d = 256 MB │
└─────────────────────────┘           └────────────────────────┘
```

**How it works:**
```
user_id "alice_92847"   → hash("alice_92847") % 1M  = bucket 742,391
user_id "bob_55123"     → hash("bob_55123")   % 1M  = bucket 103,558
user_id "carol_77610"   → hash("carol_77610") % 1M  = bucket 742,391  ← COLLISION!
```

Alice and Carol map to the same bucket → they share the same ID embedding.

**Why this is okay:**
1. The ID embedding is just ONE feature among many. The model also sees
   interest_vec, activity stats, engagement rates — all user-specific.
   So Alice and Carol are still distinguishable through other features.
2. The ID embedding captures a "rough cluster" — colliding users get
   an averaged representation, which is still useful signal.
3. With 2B users / 1M buckets = ~2000 users per bucket on average,
   but most signal comes from heavy users, and distribution is skewed.

**Reducing collision impact — Double Hashing:**
```
Instead of one hash, use two independent hash functions:

embed(user) = embed_table_A[hash_A(user_id) % 1M]
            + embed_table_B[hash_B(user_id) % 1M]   ← element-wise sum

Alice:  hash_A → bucket 742,391    hash_B → bucket 289,004
Carol:  hash_A → bucket 742,391    hash_B → bucket 617,832
                   ↑ collide                  ↑ don't collide!

Total memory: 2 × 256 MB = 512 MB (still very manageable)

Probability both hashes collide = (1/1M)² = 1 in 10¹²  → negligible
```

The **element-wise sum** of two hashed embeddings gives each user a
nearly unique composite embedding, even though each table individually
has collisions. This is a standard trick used at Meta and Google.

**Same approach applies to:**
- author_id (same 2B space)
- hashtag_id (millions of unique hashtags → hash to ~100K buckets)

---

### Dense Feature Preprocessing Pipeline

Heavy-tailed continuous features break neural nets — we must normalize.

**Strategy: Log-transform → Bucketize → Embed**

```
Example: author_follower_count

Raw values:  [50, 1200, 45000, 12000000, 500000000]
                                          ← spans 7 orders of magnitude!

Step 1 — Log transform:  log(1 + x)
             [3.9, 7.1, 10.7, 16.3, 20.0]
                                          ← much more uniform

Step 2 — Bucketize into equal-width bins (e.g., 20 bins):
             [bin_2, bin_4, bin_6, bin_9, bin_11]
                                          ← categorical now

Step 3 — Learned embedding per bin:
             bin_4 → embed(8) = [0.12, -0.34, ...]
                                          ← model learns what each range means
```

**Why bucketize instead of just normalizing (z-score)?**
- Captures **non-linear effects**: going from 100→1000 followers matters
  more than 10M→10.1M. Buckets + embeddings learn this automatically.
- Robust to outliers (all values > max_bin collapse to last bucket)
- Standard approach at Meta/Google for ranking models at scale

**Feature-specific preprocessing:**
| Feature                | Distribution      | Preprocessing                  |
|------------------------|-------------------|--------------------------------|
| follower_count         | heavy-tailed      | log → 20 buckets → embed(8)   |
| sessions_per_day       | moderate skew     | log → 10 buckets → embed(8)   |
| posts_viewed_per_day   | moderate skew     | log → 10 buckets → embed(8)   |
| post_age_minutes       | heavy-tailed      | log → 15 buckets → embed(8)   |
| early_likes_count      | heavy-tailed      | log → 15 buckets → embed(8)   |
| like_rate, comment_rate| already [0,1]     | raw (no transform needed)      |
| interest_vec           | dense, pretrained | raw 128d (already normalized)  |
| image_embed            | dense, pretrained | raw 768d (from frozen ViT)     |

### Loss Function: Sampled Softmax (InfoNCE / Contrastive Loss) ✓

**Options considered:**
```
1. Pointwise BCE:    loss = -[y·log(σ(uᵀp)) + (1-y)·log(1-σ(uᵀp))]
   → Treats pairs independently, no relative ordering signal

2. Pairwise Triplet: loss = max(0, uᵀp_neg - uᵀp_pos + margin)
   → Compares one pos vs one neg, limited context

3. Sampled Softmax:  loss = -log[ exp(uᵀp⁺/τ) / Σᵢ exp(uᵀpᵢ/τ) ]  ✓
   → Compares one positive against ALL negatives simultaneously
```

**Why Sampled Softmax / InfoNCE?**

```
For user Alice with 1 positive + K=5 negatives in a batch:

  scores:
    dot(alice, food_post⁺)     = 0.87   ← positive (engaged)
    dot(alice, sports_post⁻)   = 0.45   ← negative
    dot(alice, fashion_post⁻)  = 0.30   ← negative
    dot(alice, news_post⁻)     = 0.15   ← negative
    dot(alice, meme_post⁻)     = 0.60   ← hard negative!
    dot(alice, music_post⁻)    = 0.20   ← negative

  softmax with temperature τ=0.1:
    P(positive) = exp(0.87/0.1) / [exp(0.87/0.1) + exp(0.45/0.1) + ... ]
    loss = -log(P(positive))

  → Gradient pushes positive score UP and ALL negative scores DOWN
  → Naturally gives more gradient to hard negatives (meme_post at 0.60)
     because they contribute more to the denominator
```

**Advantages over alternatives:**
| Property                     | BCE  | Triplet | Softmax ✓ |
|------------------------------|------|---------|-----------|
| Learns relative ordering     | ✗    | ✓       | ✓         |
| Uses all negatives at once   | ✗    | ✗       | ✓         |
| Hard-negative aware          | ✗    | manual  | automatic |
| Gradient efficiency          | low  | medium  | high      |
| In-batch neg compatible      | ✗    | ✗       | ✓ perfect |

**Temperature τ controls hardness:**
- τ → small (0.05): very peaked distribution, model focuses heavily
  on hardest negatives → sharper but risk of training instability
- τ → large (1.0): softer distribution, all negatives weighted more
  equally → more stable but slower convergence
- **Sweet spot: τ ≈ 0.07-0.1** (tune on validation set)

**In-batch negatives make this efficient:**
```
Mini-batch of B=1024 (user, post⁺) pairs:
  → Each user's positive post is a negative for all other users
  → Gives B-1 = 1023 free negatives per example
  → No need to explicitly sample most negatives
  → Just compute the B×B score matrix and apply softmax per row
```

### Training: negative sampling strategy
- **Positives:** (user, post) pairs where user engaged
- **Negatives:** critical for quality — three sources:
  1. **Random negatives:** random posts from the corpus (easy negatives)
  2. **In-batch negatives:** other posts in the same mini-batch (free, efficient)
  3. **Hard negatives:** posts that were shown but NOT engaged with
     (most informative, but must sample carefully to avoid bias)
- **Ratio:** ~1 positive : 5-10 negatives (mix of easy + hard)

### Serving: decoupled inference
```
OFFLINE (batch, every few hours):          ONLINE (per request, ~5-10ms):
  Post Tower                                 User Tower
  ┌──────────────────┐                       ┌────────────────┐
  │ Score all new/    │                       │ Compute user   │
  │ recent posts      │                       │ embedding for  │
  │ → post_vec (128d) │                       │ requesting user│
  │ → index in FAISS  │                       │ → user_vec     │
  └──────────────────┘                       └───────┬────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │ ANN query       │
                                              │ FAISS top-100   │
                                              │ by dot product  │
                                              └────────────────┘
```
- Post embeddings are **precomputed** and indexed → no post tower at serving
- Only the user tower runs online → fast
- FAISS/ScaNN handles billion-scale ANN in milliseconds

### Source 3: Trending / Viral (~5% of candidates)

- Posts with abnormally high early engagement rates
- Time-decay weighted: recent viral > old viral
- Simple rule: if engagement_rate > 5× author's average → candidate
- Small quota to keep feed feeling fresh + serendipitous

### Merge + Dedup + Light Pre-rank

All three sources feed into a merge step:
1. **Dedup** — same post from multiple sources → keep once
2. **Quota mixing** — enforce source ratios (80/15/5 split, tunable)
3. **Light pre-rank** — optional logistic regression (fast!) to
   trim from ~1000 → ~500 if needed for latency budget
4. Output: **~500-1000 candidates → heavy ranker**

---

## 2. ML Objective — Multi-Task Engagement Prediction

**Approach:** Predict each engagement action separately, combine with weighted score.

### Predicted targets (per user-post pair):
| Task             | Type           | Label Source              |
|------------------|----------------|---------------------------|
| P(like)          | Binary classif | did user like post?       |
| P(comment)       | Binary classif | did user comment?         |
| P(share)         | Binary classif | did user share/send?      |
| P(save)          | Binary classif | did user save post?       |
| E(dwell_time)    | Regression     | seconds spent viewing     |

### Final Ranking Score:
```
Score = w1·P(like) + w2·P(comment) + w3·P(share) + w4·P(save) + w5·f(E(dwell))

where w3 > w2 > w4 > w1 > w5  (shares most valuable, dwell least)
```

**Why multi-task over single target?**
- Different actions have different business value → tunable weights without retraining
- Shared representation learns richer signal → each task regularizes the others
- Can adjust weights post-hoc (e.g., want more shares? increase w3)

---

## 3. Features

### 3a. User Features
- Demographics (age bucket, country, language)
- Activity level (sessions/day, posts liked/day)
- Historical engagement rates (overall like rate, comment rate)
- Preferred content type (photo vs video vs carousel)
- Account age, follower/following counts

### 3b. Post (Item) Features
- Post type (photo / video / carousel / reel)
- Caption embedding (text encoder → dense vector)
- Image/video embedding (vision model → dense vector)
- Hashtags (embedded or categorical)
- Post age (minutes since posted — very important, recency signal)
- Author follower count, posting frequency
- **Early engagement stats** (likes, comments in first N minutes — strong signal)

### 3c. User × Post Cross Features  ⭐ (highest impact)

#### (i) User-Author Affinity
How strongly does this user engage with this post's author?
Computed at multiple time windows (7d, 30d, 90d):

```
Example: Alice × @chef_mario (last 30d)
  impressions:      20
  like_rate:        15/20 = 0.75
  comment_rate:      5/20 = 0.25
  save_rate:         3/20 = 0.15
  any_engage_rate:  16/20 = 0.80   ← strong affinity

Example: Alice × @random_brand (last 30d)
  impressions:      20
  like_rate:         1/20 = 0.05
  comment_rate:      0/20 = 0.00
  save_rate:         0/20 = 0.00
  any_engage_rate:   1/20 = 0.05   ← weak affinity
```
→ New post from @chef_mario gets a massive ranking boost for Alice.

#### (ii) User-Topic Match
Match user interest vector against post topic vector.

**Build user interests:** aggregate topic distributions of posts the user
engaged with, weighted by recency + engagement type.

**Build post topic:** topic classifier or embedding clusters → distribution.

```
Alice interest vector:         New post topic:
  food:    0.60                  food:   0.9
  travel:  0.20                  travel: 0.1
  fitness: 0.15
  fashion: 0.05

Cross score = dot(user, post) = 0.6×0.9 + 0.2×0.1 = 0.56  ✓ high

Another post (fashion: 0.8, fitness: 0.2):
Cross score = 0.05×0.8 + 0.15×0.2 = 0.07  ✗ low
```

#### (iii) Social Closeness
- Does user DM this author? (binary + frequency)
- Mutual followers count
- Did user visit author's profile recently?
- Are they tagged together in posts?

These capture relationship strength *beyond* feed engagement.

### 3d. Context Features
- Time of day, day of week
- Device type (mobile / tablet)
- Session depth (1st post vs 50th — fatigue signal)
- Network quality (affects video ranking)

### Feature Computation Strategy — What runs when?

```
┌────────────────────┬──────────────┬────────────────────────────────┐
│ Feature type       │ Computed     │ How                            │
├────────────────────┼──────────────┼────────────────────────────────┤
│ User features      │ Batch        │ Daily Spark job → Feature Store│
│ (activity, rates)  │ (daily)      │ Key: user_id                  │
│                    │              │                                │
│ Post features      │ Batch        │ Hourly Spark job → Feature Str │
│ (author stats,     │ (hourly)     │ Key: post_id                  │
│  embeddings)       │              │                                │
│                    │              │                                │
│ User×Author        │ Batch        │ Daily Spark job:               │
│ affinity features  │ (daily)      │   GROUP BY (user, author)      │
│                    │              │   over interaction logs         │
│                    │              │ → Feature Store                │
│                    │              │ Key: (user_id, author_id)      │
│                    │              │                                │
│ Early engagement   │ Near-RT      │ Streaming: Kafka → Flink       │
│ (likes in 1st hr)  │ (minutes)    │ → Feature Store (hot path)     │
│                    │              │ Key: post_id                   │
│                    │              │                                │
│ Context features   │ Real-time    │ Computed in ranking server     │
│ (time, session     │ (per request)│ at request time. No storage.   │
│  depth, device)    │              │                                │
└────────────────────┴──────────────┴────────────────────────────────┘

**Scope: only compute for pairs with ≥1 impression in 90d window**
  Active users (500M) × avg ~100 active-followed authors = ~50B pairs
  Storage: 50B × 100 bytes = ~5 TB (sharded Redis, manageable)

  NOT all user×author pairs — that would be 4×10¹⁸, impossible.

**Missing pairs (no history) → fallback:**
  1. User's global engagement rates (alice likes 12% of all posts)
  2. Author's global received rates (chef_mario gets liked 8%)
  3. Binary flag: has_affinity_history = 0 (model learns to rely on
     other features when affinity is unavailable)

At serving time — single batch read from Feature Store:
  user features:     1 key lookup
  post features:     ~500 key lookups (one per candidate)
  affinity features: ~300 key lookups (one per unique author)
    → some may MISS → fill with fallback values
  ───────────────────────────────────────
  Total: ~800 keys in one batch GET → ~5-10ms from Redis
```

## 4. Model Architecture

### Baseline (v1): LightGBM per task
- One model per engagement type (like, comment, share, save, dwell)
- Input: all tabular features (affinity, counts, rates, context)
- Pros: fast training, interpretable, validates feature quality
- Cons: no embedding input, no shared learning, 5 separate models
- **No raw IDs!** Replace user_id/post_id/author_id with hand-crafted
  aggregates (engagement rates, counts, affinity scores). Trees can't
  learn embeddings — that's the key limitation vs deep models.
- Low-cardinality categoricals (post_type, device, country) are fine.
- **Use as: launch baseline + ongoing sanity-check benchmark**

### Production (v2): Multi-Task Deep Network with MMoE

```
            ┌──────────────────────── INPUT ────────────────────────┐
            │                                                       │
     Sparse Features                                        Dense Features
  (user_id, author_id,                                   (affinity scores,
   post_type, device,                                     engagement rates,
   hashtags, ...)                                          counts, time, ...)
            │                                                       │
            ▼                                                       ▼
   ┌─────────────────┐                                     ┌──────────────┐
   │  Embedding       │                                     │  Batch Norm   │
   │  Lookup Tables   │                                     │              │
   └────────┬────────┘                                     └──────┬───────┘
            │                                                      │
            └──────────────┬───────────────────────────────────────┘
                           │  concatenated vector
                           ▼
              ┌─────────────────────────────┐
              │     MMoE Layer               │
              │                              │
              │  ┌────────┐  ┌────────┐  ┌────────┐
              │  │Expert 1│  │Expert 2│  │Expert 3│  ... (N experts)
              │  │ (MLP)  │  │ (MLP)  │  │ (MLP)  │
              │  └───┬────┘  └───┬────┘  └───┬────┘
              │      │           │           │
              │      └───────────┼───────────┘
              │                  │
              │    Per-task gating networks:
              │    gate_like(x)  = softmax(W_like · x)
              │    gate_comment(x) = softmax(W_comment · x)
              │    ...each gate outputs mixture weights over experts
              │
              └─────────────────────────────┘
                           │
          ┌────────────────┼────────────────┬──────────────┐
          ▼                ▼                ▼              ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐
   │ Like Tower │  │Comment Twr │  │ Share Twr  │  │ Save Twr │ ...
   │  (MLP)     │  │  (MLP)     │  │  (MLP)     │  │  (MLP)   │
   │  → σ       │  │  → σ       │  │  → σ       │  │  → σ     │
   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────┬─────┘
         │               │               │              │
      P(like)        P(comment)       P(share)       P(save)    E(dwell)
         │               │               │              │          │
         └───────────────┴───────────────┴──────────────┴──────────┘
                                    │
                          Weighted combination
                                    │
                              Final Score
```

### Why MMoE over alternatives?
| Approach       | Pros                        | Cons                          |
|----------------|-----------------------------|-------------------------------|
| Shared-bottom  | Simple                      | Tasks forced through same repr|
| **MMoE** ✓     | Tasks learn own expert mix   | Moderate complexity           |
| PLE            | Most flexible                | Overkill for initial launch   |

### Key design decisions:
- **Embedding dim:** ~64 for user/author IDs (hashed to limit table size)
- **Experts:** 6-8 expert MLPs, each 3 layers (256→128→64)
- **Tower heads:** 2-layer MLPs (128→64→1) with sigmoid (or linear for dwell)
- **ID hashing:** user/author IDs → hash to fixed-size table (e.g., 1M buckets)
  to control memory. Trade-off: some collision but bounded memory.
- **Feature interactions:** the MLP experts learn these implicitly; optionally
  add explicit cross-layers (DCN-style) before experts for stronger signal

## 4b. MVP Rollout Strategy — Build Incrementally

```
v0: Follow-graph + heuristic score        ← START HERE
    (no ML, collects impression logs)
          │
          │  weeks of data collected
          ▼
v1: + GBDT ranker (LightGBM)             ← first ML model
    (trained on v0's impression logs)
          │
          │  months of richer data
          ▼
v2: + Deep MMoE multi-task ranker         ← production model
    (replaces GBDT, trained on v1 data)
          │
          │  stable ranking in place
          ▼
v3: + Two-tower retrieval                 ← expand beyond follows
    (suggested content from non-followed)
```

**Why heuristic v0 over two-tower v0?**
- For follow-graph content, direct affinity ≈ learned embeddings
  (we already HAVE the interaction history — no need to learn it)
- Two-tower can't use cross/context/real-time features
- Ships faster, more debuggable, fewer failure modes
- Validates infrastructure before adding ML complexity
- Two-tower is most valuable for *suggested content* (v3), not follow-graph

**Why not skip to v2?**
- Need impression logs to train ranker → need deployed system first
- Heuristic v0 provides that bootstrap data
- Each stage validates assumptions before adding complexity
- Faster time-to-production, incremental value delivery

---

## 5. Training Pipeline

### 5a. Training Data

**Source:** impression logs — every (user, post) pair that was *shown*.

```
Each training example:
┌──────────┬──────────┬───────────┬───────┬─────────┬───────┬───────┬────────────┐
│ user_id  │ post_id  │ timestamp │ liked │commented│ shared│ saved │ dwell_sec  │
├──────────┼──────────┼───────────┼───────┼─────────┼───────┼───────┼────────────┤
│ alice    │ p_001    │ 03-23 9am │   1   │    0    │   0   │   1   │    8.3     │
│ alice    │ p_002    │ 03-23 9am │   0   │    0    │   0   │   0   │    1.1     │
│ bob      │ p_001    │ 03-23 10am│   1   │    1    │   0   │   0   │   12.7     │
└──────────┴──────────┴───────────┴───────┴─────────┴───────┴───────┴────────────┘
```

**Critical rule:** only train on impressed (shown) posts.
- Positives = user saw the post AND engaged
- Negatives = user saw the post AND did NOT engage
- Never use non-shown posts — we don't know counterfactual engagement

**Scale:** ~billions of impressions per day → sample a window (e.g., last 7-14 days)

### 5b. Training Biases to Address

**Position bias:** posts shown at position 1 get more engagement than
position 10, regardless of quality.

```
Same post at different positions:
  Position 1: P(like) = 0.15
  Position 5: P(like) = 0.08   ← same post, less engagement
  Position 20: P(like) = 0.03
```

**Fix:** add position as an input feature during training, but **drop it
at inference** (set to a constant or remove). This lets the model learn
to factor out position effects. Alternatively, use a shallow position-bias
tower whose output is subtracted at serving time.

```
Training:                          Serving:
  features + position → model       features only → model
                                     (position tower removed)
```

**Selection bias:** training data was generated by the previous model,
so it's biased toward what that model liked.

**Fix:** periodically inject a small % of randomly-ranked impressions
(exploration traffic, ~1-5%) to get unbiased engagement signals.

### 5c. Loss Functions

**Per-task losses:**
```
L_like    = BCE(σ(ŷ_like),    y_like)       ← binary cross-entropy
L_comment = BCE(σ(ŷ_comment), y_comment)
L_share   = BCE(σ(ŷ_share),   y_share)
L_save    = BCE(σ(ŷ_save),    y_save)
L_dwell   = Huber(ŷ_dwell,    log(1 + y_dwell))  ← regression on log-dwell
```

**Why Huber for dwell?** Dwell time has extreme outliers (user leaves
phone open for 30 min). MSE would be dominated by outliers. Huber loss
is quadratic for small errors, linear for large → robust to outliers.
We predict log(1+dwell) to compress the range.

**Combined multi-task loss:**
```
L_total = α₁·L_like + α₂·L_comment + α₃·L_share + α₄·L_save + α₅·L_dwell
```

**Setting task weights α:** not the same as the scoring weights w.
- α weights balance *gradient magnitudes* across tasks during training
- w weights balance *business value* at inference scoring
- Start with α = 1/N (equal), then tune based on which tasks are
  under/over-fitting. Use **uncertainty weighting** (Kendall et al.)
  to learn α automatically:

```
α_task = 1 / (2·σ²_task)    where σ_task is a learned parameter

→ Tasks with higher noise (harder tasks) get lower weight automatically
→ Prevents noisy tasks (e.g., share — very sparse) from dominating gradients
```

### 5d. Class Imbalance

Engagement actions are very sparse:
```
Approximate positive rates in impression data:
  Like:     ~5%       (1 in 20 impressions)
  Comment:  ~0.5%     (1 in 200)
  Share:    ~0.2%     (1 in 500)
  Save:     ~1%       (1 in 100)
```

**Approach: downsample ALL-NEGATIVE impressions only + focal loss**

Multi-task constraint: can't downsample per-task independently
(removing a share-negative might discard a like-positive).

1. Only downsample impressions with ZERO engagement across ALL tasks
   (~85% of data → keep 30% of these, keep 100% of any-positive)
   → No positive signal lost, all tasks become more balanced
   → Training ~2.5× faster

2. Use **focal loss** per task to handle remaining imbalance:
   ```
   FL = -αₜ(1 - pₜ)^γ · log(pₜ)     (γ=2 typical)
   ```
   → Auto down-weights easy negatives (like task has many)
   → Rare-task hard examples (shares) get higher relative gradient
   → Less manual tuning than fixed class weights

3. Recalibrate at serving — **per-task, not global**:
   ```
   p_true_task = (p_model × r_task) / (p_model × r_task + (1 - p_model))

   where r_task = task-specific effective negative retention rate

   r_like  = 0.374  (more of its negatives were in other-positive pool)
   r_share = 0.404  (almost all negatives were all-negative)

   Each task has different r because downsampling all-negatives
   removes different fractions of each task's negative pool.
   ```

### 5e. Training Strategy

| Decision              | Choice                          | Rationale                         |
|-----------------------|---------------------------------|-----------------------------------|
| Optimizer             | Adam (lr=1e-3, warm-up)        | Standard for deep ranking models  |
| Batch size            | 4096-8192                       | Large batch for stable gradients  |
| Training window       | Last 14 days of impressions     | Fresh data, enough volume         |
| Retraining frequency  | Daily (full retrain)            | Keeps model fresh with trends     |
| Validation            | Next-day prediction             | Time-split, never random split    |
| Regularization        | Dropout(0.1) + L2 on embeds    | Prevent overfit on sparse IDs     |

**Time-split validation is critical:** always train on days 1-13,
validate on day 14. Never random-split impression data — that leaks
future information and inflates offline metrics.

## 6. Serving & Latency

*(to be detailed next)*

## 7. Evaluation & Metrics

*(to be detailed next)*