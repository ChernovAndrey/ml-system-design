# Ads Ranking System Design — Search Ads

## Requirements

| Dimension         | Details                                                    |
|-------------------|------------------------------------------------------------|
| Ad type           | Search ads (query-triggered, shown alongside organic results) |
| Scale             | 100M DAU, ~10-20K QPS avg, thousands of advertisers        |
| Objective         | Maximize advertiser ROI without degrading user engagement   |
| Data available    | User metadata, engagement history, ad image/text creatives  |
| Latency budget    | < 50ms end-to-end for ad serving; < 20ms for ML ranking    |

## Metrics

### Online (Business)
- **Revenue**: eCPM (effective cost per mille), revenue per search
- **Advertiser ROI**: conversion rate, ROAS (return on ad spend)
- **User engagement guardrail**: search session length, organic CTR (must not degrade)

### Model (Offline)
- **P(click)**: AUC-ROC, log-loss (calibration matters for auction!)
- **P(conversion|click)**: AUC-ROC, log-loss
- **Relevance model**: NDCG against human-labeled relevance

---

## High-Level Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AD SERVING PIPELINE                         │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │  1. RETRIEVAL │──▶│  2. RANKING  │──▶│  3. AUCTION &     │   │
│  │  (Candidate   │   │  (ML Scoring) │   │     ALLOCATION    │   │
│  │   Generation) │   │               │   │                   │   │
│  │               │   │  P(click)     │   │  Ad Rank =        │   │
│  │  Query →      │   │  P(conv|clk)  │   │   bid × P(clk)   │   │
│  │  keyword      │   │  Relevance    │   │   × quality       │   │
│  │  matching     │   │  score        │   │                   │   │
│  │               │   │               │   │  Pricing (GSP)    │   │
│  │  ~100K ads →  │   │  ~500 ads →   │   │  Budget pacing    │   │
│  │   ~500 ads    │   │   ~10-20 ads  │   │  Freq capping     │   │
│  └──────────────┘   └──────────────┘   └───────────────────┘   │
│         < 5ms              < 20ms              < 5ms            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Ad Response (ranked ads with positions + prices)
```

---

## Stage 1: Candidate Retrieval (DEEP DIVE)

**Goal**: Narrow from full inventory to ~500 relevant candidates. Must be FAST (< 5ms).

### How Advertisers Set Up Targeting

Advertisers don't say "show my ad to everyone." They create a structure:

```
Advertiser: Nike
  └─ Campaign: "Spring Running Shoes 2026"
       ├─ Ad Group: "Performance Running"
       │    ├─ Ads: [ad creative 1, ad creative 2, ...]
       │    ├─ Keywords + Match Types:
       │    │    ├─ "running shoes"         [EXACT]    bid: $2.50
       │    │    ├─ "best running shoes"    [PHRASE]   bid: $2.00
       │    │    └─ "athletic footwear"     [BROAD]    bid: $1.20
       │    └─ Targeting: US, English, Mobile+Desktop
       │
       └─ Ad Group: "Casual Sneakers"
            ├─ Ads: [ad creative 3, ...]
            ├─ Keywords:
            │    ├─ "Nike sneakers"         [EXACT]    bid: $3.00
            │    └─ "casual shoes"          [BROAD]    bid: $0.80
            └─ Targeting: US, English, Desktop only
```

Each keyword + match type = a targeting rule. The **match type** controls
how loosely the user's query can relate to the keyword and still trigger
the ad:

### Match Types Explained

User query: **"best running shoes for flat feet"**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MATCH TYPES                                 │
│                                                                     │
│  EXACT MATCH  ─────────────────────────────────────────────────     │
│  Keyword: "running shoes"                                           │
│  Matches: "running shoes" only (+ minor variants like plurals)      │
│  Query "best running shoes for flat feet" → ❌ NO MATCH             │
│  Query "running shoes"                    → ✅ MATCH                │
│                                                                     │
│  Why use: Highest intent, highest CTR, highest bid.                 │
│  Advertiser pays more because they KNOW the user wants this.        │
│                                                                     │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│  PHRASE MATCH ─────────────────────────────────────────────────     │
│  Keyword: "best running shoes"                                      │
│  Matches: query must contain the phrase (in order), can have         │
│           words before or after                                     │
│  Query "best running shoes for flat feet" → ✅ MATCH                │
│  Query "running best shoes"               → ❌ NO MATCH (wrong order)│
│                                                                     │
│  Why use: Broader reach, still high intent. Allows modifiers.       │
│                                                                     │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│  BROAD MATCH  ─────────────────────────────────────────────────     │
│  Keyword: "athletic footwear"                                       │
│  Matches: query is semantically related — doesn't need to contain   │
│           the keyword words at all                                  │
│  Query "best running shoes for flat feet" → ✅ MATCH                │
│  Query "gym sneakers"                     → ✅ MATCH                │
│  Query "hiking boots waterproof"          → ✅ MATCH (maybe)        │
│                                                                     │
│  Why use: Maximum reach, but lower precision. Lower bid.            │
│  This is where ML enters the retrieval stage! (see below)           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Inverted Index: How We Match at Query Time

We maintain an **inverted index** — the same core data structure used in
search engines for organic results, but for ad keywords:

```
INVERTED INDEX (keyword → ads)
────────────────────────────────────────────────
"running shoes"        → [(ad_101, EXACT, $2.50, Nike),
                          (ad_205, EXACT, $2.20, Adidas),
                          (ad_340, PHRASE, $1.80, Brooks), ...]

"best running shoes"   → [(ad_101, PHRASE, $2.00, Nike),
                          (ad_412, EXACT, $2.80, Hoka), ...]

"sneakers"             → [(ad_103, EXACT, $3.00, Nike),
                          (ad_520, BROAD, $1.00, Puma), ...]

"athletic footwear"    → [(ad_101, BROAD, $1.20, Nike), ...]
────────────────────────────────────────────────
```

### Query-Time Retrieval Flow

#### Step 0: Query Understanding & Normalization

Before we touch the inverted index, we need to clean up the raw query.
Users type messy things. This step runs FIRST and feeds ALL downstream
matching.

```
Raw user input
      │
      ▼
┌───────────────────────────────────────────────────────────────────┐
│                    QUERY UNDERSTANDING PIPELINE                    │
│                                                                   │
│  1. BASIC NORMALIZATION (rule-based, < 0.1ms)                     │
│     ├─ Lowercase:        "RUNNING Shoes" → "running shoes"        │
│     ├─ Strip punctuation: "running shoes!!!" → "running shoes"    │
│     ├─ Unicode normalize: "café" → "cafe" (NFKD decomposition)   │
│     ├─ Collapse whitespace: "running   shoes" → "running shoes"   │
│     └─ Remove stop words (optional, careful — "shoes for men"     │
│        vs "shoes men" loses meaning, so only remove clear noise   │
│        like "a", "the", not prepositions)                         │
│                                                                   │
│  2. SPELL CORRECTION (< 1ms)                                      │
│     ├─ "runnng shoez" → "running shoes"                           │
│     ├─ "addidas sneakers" → "adidas sneakers"                     │
│     ├─ "nkie air max" → "nike air max"                            │
│                                                                   │
│     Approach: Two-stage                                           │
│     a) Candidate generation: edit distance ≤ 2 from known terms   │
│        using a SymSpell dictionary (pre-computed, O(1) lookup)    │
│     b) Ranking: pick best correction using:                       │
│        - Language model score (bigram/trigram probability)         │
│        - Term frequency in query logs (popular queries are more   │
│          likely to be the intended query)                          │
│        - Click-through data: "did users who typed this actually   │
│          mean that?" learned from search logs                     │
│                                                                   │
│     ⚠ Important: We keep BOTH the original AND corrected query.   │
│     If the user actually meant "nkie" (a brand?), exact match     │
│     on the original still works. We use the corrected query       │
│     for phrase/broad match expansion. We can also show:           │
│     "Showing results for 'nike air max'. Search instead for       │
│     'nkie air max'?" — just like Google does.                     │
│                                                                   │
│  3. TOKENIZATION & STEMMING (< 0.1ms)                             │
│     ├─ Tokenize: "running shoes" → ["running", "shoes"]           │
│     ├─ Stem/Lemmatize: "running" → "run", "shoes" → "shoe"       │
│     │   (use for matching — index stores both original + stemmed)  │
│     └─ Generate sub-phrases for phrase matching:                   │
│        "best running shoes for flat feet" generates:               │
│        → "best running shoes", "running shoes",                    │
│          "running shoes for flat feet", "flat feet", etc.          │
│                                                                   │
│  4. LANGUAGE DETECTION & HANDLING (< 0.5ms)                        │
│     ├─ Detect language: "лучшие кроссовки" → Russian              │
│     ├─ Route to language-specific pipeline:                        │
│     │   ├─ Language-specific tokenizer (CJK needs segmentation:    │
│     │   │   "跑步鞋" → ["跑步", "鞋"] — no spaces in Chinese)     │
│     │   ├─ Language-specific spell correction dictionary           │
│     │   ├─ Language-specific stemmer (e.g., Snowball for European) │
│     │   └─ Language-specific stop word list                        │
│     │                                                              │
│     ├─ Cross-language handling options:                             │
│     │   a) Translate to English → match English keyword index       │
│     │      (simple, loses nuance)                                  │
│     │   b) Maintain per-language inverted indexes                   │
│     │      (better, advertisers set language targeting)             │
│     │   c) Language-agnostic embeddings for broad match             │
│     │      (multilingual BERT embeddings)                          │
│     │                                                              │
│     └─ For our system: option (b) as primary + (c) for broad match │
│        Advertisers specify target languages when creating campaigns │
│        → we maintain separate indexes per language                  │
│        → query language determines which index to hit               │
│                                                                   │
│  5. QUERY INTENT CLASSIFICATION (< 1ms)                            │
│     ├─ Commercial intent score: is this query transactional?       │
│     │   "buy running shoes" → high commercial intent               │
│     │   "how to tie running shoes" → low (informational)           │
│     │   "running shoes" → medium                                   │
│     │                                                              │
│     ├─ Category tagging: "running shoes" → [Footwear, Sports]      │
│     │                                                              │
│     └─ Why this matters:                                           │
│        - Low commercial intent → show FEWER ads (protect UX)       │
│        - Category tag → can pre-filter index to relevant segments  │
│        - This feeds into the quality_multiplier in the auction     │
│                                                                   │
│  Model: Lightweight classifier (fastText or small fine-tuned       │
│  model) trained on query logs with click/purchase labels.          │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
      │
      ▼
  Normalized query + metadata (language, intent, category, corrections)
  ready for index lookup
```

#### Retrieval Steps

```
User types: "best runnng shoez for flat feet"
                    │
                    ▼
       ┌────────────────────────┐
       │  0. QUERY UNDERSTANDING│  (described above)
       │                        │  → corrected: "best running shoes
       │                        │     for flat feet"
       │                        │  → lang: English
       │                        │  → intent: high commercial
       │                        │  → category: Footwear/Sports
       │                        │  → sub-phrases: ["running shoes",
       │                        │     "best running shoes", ...]
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  1. QUERY PROCESSING   │  ← normalize, tokenize, spell-correct
       │                        │     "best running shoes for flat feet"
       │  Tokens: [best,        │     → also generate: "running shoes",
       │   running, shoes,      │       "best running shoes" (sub-phrases)
       │   flat, feet]          │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  2. EXACT + PHRASE     │  ← Look up in inverted index:
       │     INDEX LOOKUP       │
       │                        │  a) Exact: "best running shoes for
       │  Time: < 1ms           │     flat feet" → probably no exact
       │                        │     keyword matches (too specific)
       │                        │
       │                        │  b) Phrase: check sub-phrases:
       │                        │     "running shoes" ⊂ query → ✅
       │                        │     "best running shoes" ⊂ query → ✅
       │                        │     → retrieves ~50-200 ads
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  3. BROAD MATCH        │  ← This is the ML-powered step:
       │     EXPANSION          │
       │                        │  Option A: Query expansion
       │  Time: 2-3ms           │    Generate synonyms/related terms:
       │                        │    "running shoes" → "jogging shoes",
       │                        │    "athletic footwear", "trainers"
       │                        │    Then look up those terms in the index.
       │                        │
       │                        │  Option B: Embedding retrieval (ANN)
       │                        │    Encode query into a dense vector.
       │                        │    Find nearest keyword embeddings
       │                        │    using approximate nearest neighbor
       │                        │    (FAISS / ScaNN).
       │                        │    → retrieves ~100-300 more ads
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  4. LIGHTWEIGHT        │  Discard ads where:
       │     FILTERS            │  ├─ budget exhausted (spent ≥ daily cap)
       │                        │  ├─ geo mismatch (ad targets UK only)
       │  Time: < 1ms           │  ├─ device mismatch (desktop-only ad)
       │                        │  ├─ ad disapproved (policy violation)
       │                        │  ├─ frequency capped (user saw it 5x)
       │                        │  └─ scheduling (ad runs weekdays only)
       │                        │
       │                        │  ~300-500 ads → ~200-500 candidates
       └───────────┬────────────┘
                   │
                   ▼
          ~200-500 candidate ads
          passed to RANKING stage
```

### Broad Match: Where Retrieval Gets Interesting

Broad match is the most nuanced tier. The advertiser bid on
"athletic footwear" but the user searched "best running shoes for
flat feet" — these share zero words but are semantically related.

**Two approaches, and I'd use both:**

**Approach A — Query Rewriting / Expansion (simpler, start here)**
- Maintain a synonym/related-terms table (built from click logs):
  if users who search "running shoes" also click on ads for keyword
  "athletic footwear", those terms are related
- At query time, expand query with top-K related terms
- Look up expanded terms in the inverted index
- **Trade-off**: cheap, interpretable, but limited to known mappings

**Approach B — Embedding-based Retrieval (more powerful, add later)**
- Pre-compute embeddings for all advertiser keywords using a
  trained query-keyword relevance model
- At query time, encode the query, do ANN search against keyword
  embeddings using FAISS or ScaNN
- Retrieve keywords with cosine similarity > threshold
- **Trade-off**: captures novel semantic relations, but adds ~2ms
  latency and needs embedding model maintenance

```
Embedding Retrieval for Broad Match:

  Query: "best running shoes for flat feet"
         │
    Encode → query_vec = [0.23, -0.11, 0.87, ...]
         │
    ANN Search (FAISS, < 2ms)
         │
    Nearest keyword embeddings:
    ├─ "running shoes"      sim=0.95  → already found by exact/phrase
    ├─ "athletic footwear"  sim=0.82  → ✅ NEW, broad match
    ├─ "jogging shoes"      sim=0.79  → ✅ NEW, broad match
    ├─ "trail running gear" sim=0.71  → ✅ NEW, broad match
    └─ "hiking boots"       sim=0.55  → below threshold, skip
```

### Index Updates: Keeping It Fresh

The inverted index isn't static — advertisers change bids, pause
campaigns, and run out of budget throughout the day:

```
NEAR-REAL-TIME UPDATES (seconds to minutes):
├─ Budget exhaustion: ad spent daily cap → remove from index
├─ Budget replenishment: new day starts → re-add to index
├─ Bid changes: advertiser adjusts bid → update index entry
├─ Ad pause/resume: advertiser toggles campaign → add/remove
└─ New ads: after approval → add to index

BATCH UPDATES (hourly/daily):
├─ Broad match embedding recomputation
├─ Synonym table refresh from click logs
└─ Quality score recalculation
```

**Implementation**: The index lives in memory across sharded servers.
Updates propagate via a message queue (Kafka). Each shard holds a
portion of the keyword space. Query fanout hits relevant shards in
parallel, results are merged.

### Why Not Skip Retrieval and Score Everything?

With thousands of advertisers but maybe ~100K-1M active ad-keyword
pairs, one might ask: can we just score them all?

**No**, because:
- Even at 0.1ms per ad (LR), scoring 1M ads = 100 seconds. Way over budget.
- The DNN at ~0.02ms per ad (batched) × 1M = 20 seconds. Still way over.
- Retrieval narrows to ~500 candidates, making ranking feasible:
  500 × 0.02ms = 10ms ✅

Retrieval is about **recall** — we want to make sure every
*plausibly relevant* ad makes it through. The ranking model then
handles **precision** — ordering them correctly.

---

## Stage 2: Ranking (ML Scoring) ← THE CORE

### Baseline: Logistic Regression

Before going complex, a **logistic regression** on hand-crafted features gives us:
- Fast inference (< 1ms for 500 candidates)
- Interpretable coefficients (debuggable, explainable to advertisers)
- Well-calibrated probabilities (critical for auction pricing)
- Features: query-ad keyword overlap, historical CTR of ad, user CTR, position bias

**Why a SINGLE LR for P(click) only — not three models:**

```
Multi-task makes sense for DNN        For LR baseline, single task:
(shared embeddings help sparse         ┌─────────────────────────────┐
 tasks learn from dense ones)          │   LR predicts P(click) ONLY │
                                       │                             │
 NOT this for LR:                      │   Why:                      │
 ┌───────┐ ┌──────────┐ ┌──────┐      │   1. P(click) has the most  │
 │P(clk) │ │P(conv|c) │ │Relev.│      │      training data — every  │
 │  LR   │ │   LR     │ │  LR  │      │      impression is a sample │
 └───────┘ └──────────┘ └──────┘      │   2. P(conv|click) is too   │
 3 separate models,                    │      sparse for LR — only   │
 no shared learning,                   │      clickers give labels,  │
 conversion LR starved                 │      conversions are rare   │
 for data                              │   3. Relevance is captured  │
                                       │      by text features (BM25,│
                                       │      Jaccard) already in    │
                                       │      the click model        │
                                       └─────────────────────────────┘
```

**Baseline auction formula with LR:**
```
Ad Rank = bid × P(click)_LR × quality_heuristic

where quality_heuristic = rule-based score from:
  ├─ keyword_match_type weight (exact=1.2, phrase=1.0, broad=0.8)
  ├─ landing_page_load_time score (fast=1.0, slow=0.7)
  └─ advertiser_historical_quality (CTR percentile bucket)
```

This is good enough to launch, establish baseline metrics, and
start collecting the conversion data we'll need for the DNN.

**When we graduate to DNN → multi-task becomes valuable:**
- Shared embedding layers let the sparse conversion head learn
  from the dense click signal (knowledge transfer)
- The relevance head can learn from human-labeled data and
  feed into a learned quality score (replacing the heuristic)
- All three outputs are used in a richer auction formula

### Production Model: Deep Ranking Model

```
┌─────────────────────────────────────────────────────┐
│              DEEP RANKING MODEL                      │
│                                                      │
│  Inputs:                                             │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Query   │ │ User     │ │ Ad       │ │ Context │ │
│  │ Features│ │ Features │ │ Features │ │ Features│ │
│  └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬────┘ │
│       │           │            │             │      │
│       ▼           ▼            ▼             ▼      │
│  ┌─────────────────────────────────────────────┐    │
│  │         Feature Interaction Layer            │    │
│  │   (Crosses, embeddings, attention)           │    │
│  └──────────────────┬──────────────────────────┘    │
│                     │                                │
│          ┌──────────┼──────────┐                     │
│          ▼          ▼          ▼                     │
│     ┌────────┐ ┌────────┐ ┌─────────┐               │
│     │P(click)│ │P(conv  │ │Relevance│               │
│     │  head  │ │ |click)│ │  head   │               │
│     └────────┘ └────────┘ └─────────┘               │
│                                                      │
│  Architecture: Two-tower with cross-attention        │
│  (similar to DLRM / DCN-v2 patterns)                │
└─────────────────────────────────────────────────────┘
```

### Feature Groups

| Group       | Features                                                           |
|-------------|--------------------------------------------------------------------|
| **Query**   | Query text embedding, query length, query category/intent          |
| **User**    | User embedding (from history), demographics, device, geo           |
| **Ad**      | Ad text embedding, historical CTR, advertiser quality, landing page|
| **Context** | Time of day, day of week, position, device type                    |
| **Cross**   | Query-ad text similarity, user-ad category affinity, past clicks   |

### Multi-task Heads
- **P(click)** — primary signal, most training data
- **P(conversion | click)** — higher value but sparser, uses click data
- **Relevance score** — trained on human-judged query-ad relevance labels
- Shared bottom layers allow knowledge transfer; separate heads avoid
  task interference.

---

### DEEP DIVE: Feature Processing

The fundamental difference: **LR needs you to engineer the signal;
DNN can learn from rawer representations but still needs thoughtful input.**

#### Raw Data We Collect Per Request

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                             │
│                                                                     │
│  REQUEST TIME:                    PRE-COMPUTED (Feature Store):     │
│  ├─ query string: "best running   ├─ user profile                  │
│  │   shoes for flat feet"          │   ├─ age_bucket, gender, geo   │
│  ├─ user_id: u_12345              │   ├─ click_history (last 90d)  │
│  ├─ device: iPhone 14             │   └─ category_preferences      │
│  ├─ timestamp: 2026-03-22 14:30   │                                │
│  ├─ session context:              ├─ ad metadata                   │
│  │   └─ previous queries in       │   ├─ ad_id, campaign_id        │
│  │      session                    │   ├─ ad title, description     │
│  └─ geo: San Francisco            │   ├─ landing page URL          │
│                                    │   ├─ historical CTR, CVR       │
│                                    │   └─ advertiser quality score  │
│                                    │                                │
│                                    ├─ keyword metadata              │
│                                    │   ├─ matched keyword text      │
│                                    │   ├─ match type (exact/broad)  │
│                                    │   └─ keyword-level CTR         │
│                                    └─ bid: $2.50 CPC               │
└─────────────────────────────────────────────────────────────────────┘
```

#### Feature Processing for Logistic Regression

LR can only learn **linear** decision boundaries. So every useful
interaction and non-linearity must be **hand-crafted** as a feature.

```
RAW DATA ──▶ FEATURE ENGINEERING ──▶ FEATURE VECTOR ──▶ LR MODEL
                                     (all numeric)
```

**1. Text Matching Features (query ↔ ad)**
These are the most important features for search ads — they capture relevance.

| Feature                    | How computed                                             | Why it matters                        |
|----------------------------|---------------------------------------------------------|---------------------------------------|
| exact_keyword_match        | 1 if query == bid keyword exactly                        | Strongest relevance signal            |
| keyword_match_type         | one-hot: exact / phrase / broad                          | Exact match → higher intent alignment |
| query_ad_title_overlap     | Jaccard(query_tokens, ad_title_tokens)                   | Token-level text similarity           |
| query_ad_desc_overlap      | Jaccard(query_tokens, ad_desc_tokens)                    | Same for description                  |
| BM25_score                 | BM25(query, ad_text)                                     | TF-IDF style relevance, proven in IR  |
| query_keyword_char_edit    | Normalized edit distance(query, keyword)                 | Catches typos, close matches          |
| word_coverage_ratio        | % of query words appearing in ad text                    | Penalizes ads that miss key terms     |

**2. Historical Performance Features (statistical aggregates)**
These encode what we've *already observed* about how entities perform.

| Feature                    | How computed                                              | Why it matters                       |
|----------------------------|---------------------------------------------------------  |--------------------------------------|
| ad_historical_CTR          | smoothed clicks/impressions for this ad (last 30d)        | Best predictor of future clicks      |
| keyword_historical_CTR     | smoothed CTR for this keyword across all ads              | Captures keyword-level intent        |
| advertiser_avg_CTR         | avg CTR across all ads from this advertiser               | Advertiser quality signal            |
| user_historical_CTR        | this user's overall click rate on ads                     | Some users click more than others    |
| ad_CTR_on_this_keyword     | CTR of this specific (ad, keyword) pair                   | Most specific but sparsest signal    |
| ad_conversion_rate         | conversions/clicks for this ad (last 30d)                 | Key for ROI objective                |

**Smoothing is critical**: raw CTR for a new ad with 10 impressions is
unreliable. Use **empirical Bayes smoothing**:
`smoothed_CTR = (clicks + α * global_CTR) / (impressions + α)`
where α is a prior weight (~200-1000 impressions). This pulls low-data
entities toward the global mean.

**3. User Features**

| Feature                    | How computed                                             |
|----------------------------|---------------------------------------------------------|
| user_ad_category_affinity  | user's CTR on ads in this ad's category                  |
| user_device_type           | one-hot: mobile / desktop / tablet                       |
| user_geo_bucket            | geo cluster (city-level too sparse → region buckets)     |
| user_session_depth         | # queries so far in this session                         |
| user_recency               | days since last ad click                                 |

**4. Context Features**

| Feature                    | How computed                                             |
|----------------------------|---------------------------------------------------------|
| hour_of_day                | bucketed (morning/afternoon/evening/night)               |
| day_of_week                | one-hot (7 values)                                       |
| is_weekend                 | binary                                                   |
| ad_position_slot           | which slot on the page (used in TRAINING only — see note)|

**5. Explicit Feature Crosses (THIS IS KEY FOR LR)**

LR cannot learn interactions, so we create them manually:

| Cross Feature                       | Why                                                    |
|-------------------------------------|--------------------------------------------------------|
| match_type × ad_historical_CTR      | exact matches with high CTR → very strong signal       |
| device_type × ad_historical_CTR     | mobile vs desktop CTR patterns differ significantly    |
| user_category_affinity × ad_category| does this user like this type of ad?                   |
| hour_bucket × ad_CTR                | some ads perform differently at different times        |
| query_length × keyword_match_type   | long queries on broad match → lower relevance          |

**Total LR feature vector: ~50-100 dimensions** (after one-hot encoding)

```
LR Input Vector (for one query-ad pair):
[exact_match, phrase_match, broad_match, jaccard_title=0.4,
 jaccard_desc=0.2, BM25=12.3, ad_CTR=0.035, kw_CTR=0.028,
 user_CTR=0.041, ..., match_type_x_adCTR=0.035, ...]
     │
     ▼
 w · x + b → σ(z) → P(click) = 0.037
```

---

#### Feature Processing for DNN (DCN-v2)

The DNN changes the game: we can feed **rawer features** and let the
network learn interactions. But we still need thoughtful input design.

```
RAW DATA ──▶ EMBEDDING + NORMALIZATION ──▶ DNN ──▶ P(click)
              (per feature type)
```

**Key difference from LR: we DON'T need to hand-craft crosses.**
The Cross Network in DCN-v2 learns arbitrary feature interactions
automatically. But we DO need to represent features properly.

```
┌─────────────────────────────────────────────────────────────────┐
│                  DNN FEATURE PROCESSING                         │
│                                                                 │
│  SPARSE CATEGORICAL          DENSE NUMERICAL    TEXT            │
│  ┌──────────────────┐       ┌──────────────┐   ┌────────────┐  │
│  │ user_id          │       │ ad_hist_CTR  │   │ query text │  │
│  │ ad_id            │       │ BM25_score   │   │ ad title   │  │
│  │ keyword_id       │       │ user_CTR     │   │ ad desc    │  │
│  │ advertiser_id    │       │ session_depth│   └─────┬──────┘  │
│  │ ad_category      │       │ bid_amount   │         │         │
│  │ geo_region       │       └──────┬───────┘   Pre-trained     │
│  │ device_type      │              │           text encoder     │
│  │ match_type       │         Log-transform    (distilled BERT │
│  └────────┬─────────┘         + normalization  or fastText)    │
│           │                        │                 │         │
│     Embedding tables          Batch norm         Fixed-dim     │
│     (learned, dim=16-64)           │             embedding     │
│           │                        │                 │         │
│           ▼                        ▼                 ▼         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CONCATENATED EMBEDDING VECTOR               │   │
│  │              (~200-500 dimensions)                        │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│              ┌────────────▼───────────────┐                     │
│              │    Cross Network (DCN-v2)  │                     │
│              │    Learns: query × ad_cat, │                     │
│              │    user × device × hour,   │                     │
│              │    etc. AUTOMATICALLY       │                     │
│              └────────────┬───────────────┘                     │
│                           │                                     │
│              ┌────────────▼───────────────┐                     │
│              │    Deep Network (MLP)      │                     │
│              │    256 → 128 → 64          │                     │
│              └────────────┬───────────────┘                     │
│                           │                                     │
│                    Multi-task heads                              │
└─────────────────────────────────────────────────────────────────┘
```

**Sparse Categorical Features → Embedding Tables**

This is the biggest unlock over LR. Instead of one-hot encoding ad_id
(millions of values → millions of weights in LR, infeasible), we learn
a dense embedding:

- `ad_id` → 64-dim embedding (learned during training)
- `user_id` → 64-dim embedding
- `keyword_id` → 32-dim embedding
- `ad_category` → 16-dim embedding
- `device_type` → 8-dim embedding (small vocabulary)

These embeddings capture **latent properties**: ads with similar click
patterns will have similar embeddings, even without explicit features.

**Dense Numerical Features → Log-transform + Normalization**

Features like CTR, impressions count, bid amount have skewed distributions.

```
raw ad_impressions: [1, 5, 100, 50000, 1M]  ← huge range
log-transformed:   [0, 1.6, 4.6, 10.8, 13.8] ← manageable
then batch-normalize to mean=0, std=1
```

We STILL feed the hand-crafted stats (ad_CTR, BM25) — the DNN benefits
from these too. It just doesn't DEPEND on them as much as LR does.

**Text Features → Pre-trained Encoder**

For query and ad text, two options:
- **Fast path**: Pre-computed embeddings from a distilled BERT (6-layer),
  stored in feature store. Lookup at serving time, no inference cost.
- **Slow path**: fastText or TF-IDF embeddings. Simpler, faster, less powerful.

I'd go with **pre-computed distilled BERT embeddings** for the DNN.
The key: we compute these OFFLINE and cache them, so there's no
latency cost at serving time. New ads get their embeddings computed
when they're created.

**What the DNN learns that LR can't:**
- `user_embedding × ad_embedding` → personalized affinity
- Non-linear effects: CTR matters differently at different ranges
- High-order interactions: (query_type × device × time × ad_category)
- Long-tail generalization: new ads get useful scores from shared embeddings

---

#### LR vs DNN: When to Use Which

```
                    LR                          DNN (DCN-v2)
                    ──                          ───────────
Features needed:    ~50-100 hand-crafted        ~20 raw + embeddings
Feature crosses:    Manual (5-10 key ones)      Learned automatically
Inference speed:    < 0.1ms per candidate       ~5-10ms for 500 candidates (batched)
Calibration:        Naturally well-calibrated   Needs post-hoc calibration
New ads:            Relies on keyword-level      Embedding generalizes
                    stats (cold start harder)    (cold start better)
AUC improvement:    Baseline                    +2-5% AUC over LR typically
Debuggability:      Inspect coefficients        Black box, need SHAP/LIME
Training:           Minutes                     Hours (GPU)
```

**My recommendation**: Start with LR as v1 (ship in weeks, establish
baseline metrics), then layer in the DNN. Keep LR running as a
calibration reference and fallback.

---

### Model Architecture Choice: Why DCN-v2 / DLRM style?
- Efficient on sparse categorical features (user_id, ad_id, keyword_id)
- Explicit feature crosses capture interactions (query × ad keyword)
- Proven at scale at Meta, Google for CTR prediction
- **Trade-off**: I'm choosing this over a pure Transformer because
  latency is tight (< 20ms) and we have mostly tabular + sparse features.
  If we had richer text understanding needs, I'd consider a small
  distilled BERT for query-ad matching as a sub-component.

---

## Stage 3: Auction & Allocation

### Ad Rank Computation
```
Ad Rank = bid × P(click) × P(conversion|click) × quality_multiplier
```

- **bid**: advertiser's max CPC or CPA bid
- **P(click)**: predicted click probability from ranking model
- **quality_multiplier**: f(relevance_score, landing_page_quality, ad_format)
  - This is the lever that protects user experience!
  - Low-relevance ads get penalized even if they bid high

### Pricing: Generalized Second Price (GSP)

**Core idea**: You pay the MINIMUM needed to keep your position —
not what you bid. This means advertisers can bid their true value
without fear of overpaying.

#### Concrete Example

Three advertisers competing for query "running shoes":

```
STEP 1: Each advertiser has a bid and a quality score
─────────────────────────────────────────────────────
Advertiser    Bid (CPC)    P(click)    Quality    Ad Rank
                                       Score      = Bid × P(click) × Quality
─────────────────────────────────────────────────────
Nike          $2.00        0.08        1.2        $2.00 × 0.08 × 1.2 = 0.192
Adidas        $3.00        0.05        0.9        $3.00 × 0.05 × 0.9 = 0.135
Brooks        $1.50        0.06        1.0        $1.50 × 0.06 × 1.0 = 0.090
─────────────────────────────────────────────────────

STEP 2: Rank by Ad Rank (descending)
─────────────────────────────────────
Position 1 (top):  Nike     Ad Rank = 0.192
Position 2:        Adidas   Ad Rank = 0.135
Position 3:        Brooks   Ad Rank = 0.090
─────────────────────────────────────

Note: Adidas bid the MOST ($3.00) but ranks #2 because their
quality score and P(click) are lower. The auction rewards
relevance, not just money.

STEP 3: Pricing — what does each advertiser actually PAY per click?
──────────────────────────────────────────────────────────────────

Formula:  price = (Ad Rank of next ad below) / (your quality score) + $0.01
          ─────────────────────────────────────────────────────────

Nike (Position 1):
  price = Ad Rank of Adidas / (Nike's P(click) × Nike's quality) + $0.01
  price = 0.135 / (0.08 × 1.2) + $0.01
  price = 0.135 / 0.096 + $0.01
  price = $1.41 + $0.01
  price = $1.42 per click
                   │
                   └─ Nike BID $2.00 but only PAYS $1.42
                      They save $0.58 per click!

Adidas (Position 2):
  price = Ad Rank of Brooks / (Adidas's P(click) × Adidas's quality) + $0.01
  price = 0.090 / (0.05 × 0.9) + $0.01
  price = 0.090 / 0.045 + $0.01
  price = $2.00 + $0.01
  price = $2.01 per click
                   │
                   └─ Adidas BID $3.00 but only PAYS $2.01
                      Despite bidding highest, they pay less!
                      But they still pay MORE than Nike because
                      their quality is lower — the quality score
                      acts as a discount for good ads.

Brooks (Position 3, last):
  price = minimum bid threshold (e.g., $0.10) + $0.01
  price = $0.11 per click
                   │
                   └─ No one below them, so they pay the floor.
```

#### Why This Design Is Smart

```
┌──────────────────────────────────────────────────────────────┐
│                  WHY GSP WORKS                                │
│                                                              │
│  1. ADVERTISERS DON'T OVERPAY                                │
│     You bid $2.00 but pay $1.42. You never pay more          │
│     than necessary to hold your position. This encourages    │
│     truthful bidding — just bid what a click is worth to you.│
│                                                              │
│  2. QUALITY IS REWARDED                                      │
│     Nike bid LESS than Adidas ($2 vs $3) but ranks higher    │
│     AND pays less per click. Why? Better P(click) and        │
│     quality score. This means:                               │
│     - Relevant ads get cheaper clicks (incentivizes quality) │
│     - Irrelevant ads pay a "tax" for poor user experience    │
│     - Users see better ads → engagement stays high           │
│     → This directly serves our objective: ROI without        │
│       degrading user engagement!                             │
│                                                              │
│  3. REVENUE IS PROTECTED                                     │
│     The $0.01 increment ensures we always charge slightly    │
│     more than the minimum. And because quality multiplies    │
│     into Ad Rank, the system naturally surfaces ads that     │
│     generate more clicks → more revenue even at lower CPCs.  │
│                                                              │
│  4. WHY "SECOND PRICE" NOT "FIRST PRICE"?                    │
│     First price: you pay what you bid → everyone shades      │
│     bids down → unstable, game-theoretic nightmare.          │
│     Second price: you pay what #2 bid (adjusted) →           │
│     dominant strategy is to bid truthfully.                   │
│                                                              │
│  ⚠ Note: Google moved to first-price for display ads in      │
│  2019, but search ads still use a GSP-like mechanism.        │
│  The trade-off: first-price is simpler but requires          │
│  bid shading algorithms on the advertiser side.              │
└──────────────────────────────────────────────────────────────┘
```

### Cold Start Problem (DEEP DIVE)

Cold start affects FOUR entities — each needs a different strategy:

```
┌──────────────────────────────────────────────────────────────┐
│                WHO HAS COLD START?                            │
│                                                              │
│  1. NEW AD         — no impressions, no CTR, no embedding    │
│  2. NEW ADVERTISER — no quality history, no trust signal     │
│  3. NEW USER       — no click profile, no preferences        │
│  4. NEW KEYWORD    — no keyword-level CTR stats              │
│                                                              │
│  Impact: model relies heavily on historical features          │
│  (ad_CTR, user_CTR, etc.) — these are all ZERO/MISSING       │
│  for new entities → predictions are unreliable                │
└──────────────────────────────────────────────────────────────┘
```

#### 1. New Ads (most common, most critical)

An advertiser creates a new ad creative. We have the ad text,
landing page, bid, and targeting — but zero impressions or clicks.

**Layer A: Features that work WITHOUT history**

Even with no clicks, we still have useful signals:

```
FOR LOGISTIC REGRESSION:                FOR DNN:
├─ BM25(query, ad_text) ✅              ├─ Ad text BERT embedding ✅
├─ Keyword match type ✅                ├─ Keyword embedding ✅
├─ Advertiser-level CTR ✅              ├─ Advertiser embedding ✅
│  (fall back from ad → advertiser)     │  (shared across all their ads)
├─ Keyword-level CTR ✅                 ├─ Category embedding ✅
│  (other ads on this keyword)          └─ All context features ✅
├─ Category avg CTR ✅
└─ Landing page quality score ✅        ad_id embedding → ❌ UNKNOWN
                                        ad historical CTR → ❌ ZERO
```

The key insight: **design a feature fallback hierarchy** so we
always have something to use:

```
FEATURE FALLBACK CHAIN (most specific → most general):

ad_CTR (this specific ad)
  │ missing?
  ▼
ad_group_CTR (other ads in same ad group — same keyword targets)
  │ missing?
  ▼
advertiser_CTR (all ads from this advertiser)
  │ missing?
  ▼
keyword_CTR (all ads competing on this keyword)
  │ missing?
  ▼
category_CTR (all ads in this category, e.g., "Footwear")
  │ missing?
  ▼
global_CTR (overall platform average, ~2-3%)
```

This is essentially **hierarchical empirical Bayes smoothing**:
at each level, we blend the entity's own data with the prior from
the level above. Even a brand-new ad from an established advertiser
gets a reasonable estimate from the advertiser-level stats.

**Layer B: Exploration — giving new ads a chance**

Even with good fallback features, predictions for new ads are
UNCERTAIN. We need to show them to learn. But we can't just throw
untested ads into top positions — that risks bad user experience.

```
EXPLORATION STRATEGIES:

┌─────────────────────────────────────────────────────────┐
│  Strategy 1: EXPLORATION BUDGET (recommended, start here)│
│                                                         │
│  Reserve ~5-10% of ad impressions for "exploration."    │
│  In this slice:                                         │
│  - New ads get boosted into consideration set           │
│  - Shown in LOWER positions (position 3-4, not top)     │
│  - Still must pass minimum relevance threshold           │
│                                                         │
│  After ~1000 impressions, the ad has enough data to      │
│  compete on its own merits.                              │
│                                                         │
│  Trade-off: simple, controllable. Burns ~5% of revenue   │
│  on exploration, but this investment pays off when we     │
│  discover high-performing new ads.                       │
├─────────────────────────────────────────────────────────┤
│  Strategy 2: THOMPSON SAMPLING (more sophisticated)      │
│                                                         │
│  Model P(click) as a Beta distribution per ad:           │
│  - New ad: Beta(α=1, β=50) → wide uncertainty           │
│  - Established ad: Beta(α=350, β=9650) → tight around 3.5% │
│                                                         │
│  At serving time, SAMPLE from each ad's distribution:    │
│  - New ad might sample anywhere from 0.5% to 8%         │
│  - Established ad samples close to 3.5% every time       │
│                                                         │
│  Effect: new ads occasionally get "lucky" high samples   │
│  and win positions → get shown → we observe clicks →     │
│  distribution narrows → converges to true CTR.           │
│                                                         │
│  Trade-off: mathematically principled (balances           │
│  explore/exploit optimally), but harder to debug and     │
│  explain to advertisers. "Why did my competitor's new    │
│  ad show above mine?"                                    │
├─────────────────────────────────────────────────────────┤
│  Strategy 3: OPTIMISTIC INITIALIZATION                   │
│                                                         │
│  Give new ads a slightly HIGHER initial CTR estimate     │
│  than the prior (e.g., category_avg × 1.2).             │
│                                                         │
│  Effect: new ads get more initial impressions.           │
│  As real data comes in, the estimate corrects downward   │
│  if the ad is actually bad.                              │
│                                                         │
│  Trade-off: simplest to implement. Risk: bad ads get     │
│  shown too much before correction. Mitigate with quick   │
│  suppression if CTR is significantly below threshold     │
│  after N impressions.                                    │
└─────────────────────────────────────────────────────────┘
```

**My recommendation**: Start with **exploration budget + optimistic
initialization** (simple, controllable). Move to **Thompson sampling**
once we have the infrastructure and want to optimize exploration
efficiency.

**Layer C: DNN-specific — embedding for new ad_ids**

The DNN's ad_id embedding is unknown for a new ad. Options:

```
Option 1: Average embedding initialization
  new_ad_embedding = mean(all ad embeddings in same category)
  → reasonable starting point, refine with online updates

Option 2: Content-based embedding
  Use a SEPARATE small model that predicts ad_id embedding
  FROM ad content features (text embedding, category, etc.):

  ad_text_BERT_emb + category + landing_page → MLP → predicted ad_id_emb

  Trained on existing ads where we know both content and learned embedding.
  → For new ads, we predict what the embedding SHOULD be from content alone.

Option 3: Hash-based feature (no per-ad embedding)
  Instead of ad_id, use feature hashing on (advertiser_id, keyword,
  ad_category) → share embedding buckets across similar ads.
  New ads automatically map to relevant buckets.
  → Loses per-ad specificity but eliminates cold start entirely.
```

I'd use **Option 2** — it's the best trade-off. We get a reasonable
embedding from day one, and it refines as real data flows in.

---

#### 2. New Advertisers

Harder than new ads because we lack even the advertiser-level fallback.

```
Approach:
├─ Fall back to category_CTR and global_CTR in the hierarchy
├─ Require new advertisers to go through a "learning period":
│   ├─ First 7 days: limited impression share (~10% of what
│   │   an established advertiser would get)
│   ├─ Increase as data accumulates and quality proves out
│   └─ If quality is poor after learning period → restrict further
├─ Landing page quality pre-scan:
│   ├─ Before any impressions, crawl and score landing pages
│   ├─ Slow load, spammy content, mismatched topic → lower quality
│   └─ This gives us SOME quality signal before a single impression
└─ Industry benchmarks as prior:
    └─ "E-commerce footwear advertisers" avg CTR = 3.2% → use as prior
```

#### 3. New Users

Least problematic — we actually have a lot to work with even for
anonymous/new users:

```
Signals available for new users:
├─ Query itself (strongest signal in search ads!)
├─ Device type, browser, OS
├─ Geo (IP → city/region)
├─ Time of day, day of week
├─ Referrer / entry point
└─ Session behavior (builds up FAST — after 2-3 queries in a
   session, we already see patterns)

Approach:
├─ Context features are always available — the model doesn't
│  ONLY rely on user history
├─ User_id embedding → use average embedding (like ads)
├─ Real-time session features update within the session:
│  "user clicked 2 shoe ads this session" → strong signal
└─ After 1-2 sessions, user has enough history for
   personalization to kick in
```

New users are the least concerning because **in search ads, the
query carries most of the intent signal, not the user profile**.
This is different from feed/display ads where personalization
matters much more.

#### 4. New Keywords

When an advertiser bids on a keyword no one has used before:

```
Approach:
├─ Text similarity to existing keywords:
│  "trail running shoes" is close to "running shoes"
│  → borrow stats from similar keywords
├─ Embedding-based nearest-neighbor:
│  new_keyword_emb → find K nearest known keywords → avg their CTR
└─ Falls back to category/global CTR otherwise
```

---

#### Cold Start Summary

```
Entity           Primary Strategy         Fallback           Timeline to
                                                             "warm"
──────────────────────────────────────────────────────────────────────
New Ad           Feature hierarchy +       Category/global   ~1K impressions
                 exploration budget        CTR prior          (~1-3 days)

New Advertiser   Learning period +         Industry          ~7-14 days
                 landing page pre-scan     benchmarks

New User         Context features +        Query intent is   ~1-2 sessions
                 session signals           primary signal     (minutes)

New Keyword      Similar keyword           Category/global   ~500 impressions
                 borrowing                 CTR prior          (~1-7 days)
──────────────────────────────────────────────────────────────────────
```

### Business Logic
- **Budget pacing**: spread daily budget evenly (don't exhaust by noon)
- **Frequency capping**: max N impressions per user per ad per day
- **Policy filters**: blocked advertisers, sensitive categories, etc.

---

## Training Pipeline

```
Impression/Click/Conversion Logs
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│  Feature Store   │────▶│  Training Data   │
│  (offline)       │     │  Generation      │
└──────────────────┘     └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Model Training  │
                         │  (daily retrain) │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Validation &    │
                         │  Calibration     │ ← critical for pricing!
                         │  Check           │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Shadow / Canary │
                         │  Deployment      │
                         └──────────────────┘
```

### Key Training Considerations
- **Label**: click (1/0), conversion (1/0), with impression as negative
- **Position bias correction**: users click top positions more regardless of relevance.
  Use position as a feature in training, remove at inference. Or use causal methods.
- **Delayed conversions**: conversions can happen hours/days after click.
  Use attribution window (e.g., 7-day) with label correction for recent data.
- **Calibration**: model probabilities MUST be well-calibrated — they directly
  determine pricing. Use Platt scaling or isotonic regression post-hoc.
- **Freshness**: daily retraining + real-time feature updates (recent CTR)

---

## Serving Architecture

```
┌────────────┐       ┌──────────────┐      ┌──────────────┐
│  Feature   │       │  Model       │      │  Ads Index   │
│  Store     │       │  Server      │      │  (inverted   │
│  (Redis/   │       │  (TF Serving │      │   index)     │
│   online)  │       │   / ONNX)    │      │              │
└─────┬──────┘       └──────┬───────┘      └──────┬───────┘
      │                     │                     │
      └─────────┬───────────┘─────────────────────┘
                │
         ┌──────▼──────┐
         │  Ad Server  │  ← orchestrates the full pipeline
         │  (< 50ms)   │
         └─────────────┘
```

- **Online feature store** (Redis/Memcached): pre-computed user features,
  real-time ad CTR stats, updated in near-real-time
- **Model server**: serves ranking model, optimized with ONNX / TensorRT
  for low-latency batch inference
- **Ads index**: inverted index of keyword → ads, updated when advertisers
  modify campaigns

---

## Evaluation & Deployment

### Offline Evaluation
- AUC-ROC and log-loss on held-out test set (next-day data)
- Calibration plots (predicted vs actual CTR in buckets)
- NDCG for ranking quality

### Online A/B Testing
- **Primary metrics**: revenue per search, advertiser conversion rate
- **Guardrail metrics**: organic CTR, session length, searches per user
  (must not degrade — per our requirement!)
- Ramp: 1% → 5% → 20% → 100% over ~2 weeks
- Statistical significance: use sequential testing to detect regressions early

### Monitoring
- **Model health**: prediction distribution drift, feature drift
- **Latency**: p50/p95/p99 of each pipeline stage
- **Business**: hourly revenue, CTR, conversion rate dashboards
- **Alerts**: if calibration degrades > threshold → auto-rollback

---

## Feedback Loops, Bias & Failure Modes

### Position Bias (the most insidious problem)

Ads shown in position 1 get clicked more *regardless of quality* —
simply because users look there first. This creates a vicious cycle:

```
Ad shown in position 1
    → gets more clicks (because position, not quality)
        → model learns "this ad has high CTR"
            → model ranks it higher
                → shown in position 1 again
                    → REINFORCING LOOP ♻️

Meanwhile, a BETTER ad stuck in position 3:
    → gets fewer clicks (because position)
        → model learns "this ad has low CTR"
            → ranks it lower
                → NEVER gets a fair chance
```

**How to correct:**

```
┌──────────────────────────────────────────────────────────────┐
│  APPROACH 1: Position as training feature, removed at serving│
│                                                              │
│  Training: include position as an input feature              │
│    model learns: P(click | ad, user, query, position=1)      │
│                                                              │
│  Serving: set position = default constant for ALL ads         │
│    model predicts: P(click | ad, user, query, position=avg)  │
│    → isolates the ad's INTRINSIC quality from position effect│
│                                                              │
│  ✅ Simple, effective. Industry standard (Google, Meta).      │
│  ⚠ Assumes position effect is separable (mostly true).       │
├──────────────────────────────────────────────────────────────┤
│  APPROACH 2: Inverse propensity weighting (IPW)              │
│                                                              │
│  Weight each training example by 1/P(examined | position):   │
│  - Position 1 clicks get weight 1/0.9 = 1.1 (almost always  │
│    examined, so click is less informative)                    │
│  - Position 4 clicks get weight 1/0.3 = 3.3 (rarely examined │
│    so a click is strong evidence of quality)                  │
│                                                              │
│  Estimate examination probabilities from randomization        │
│  experiments or eye-tracking studies.                         │
│                                                              │
│  ✅ Theoretically principled (causal inference).              │
│  ⚠ High variance with extreme weights. Needs clipping.       │
└──────────────────────────────────────────────────────────────┘
```

I'd use **Approach 1** — it's proven, simple, and works well in practice.

### Rich-Get-Richer Problem

Beyond position bias, there's a broader feedback loop: advertisers
with big budgets get more impressions → more data → better model
predictions → rank higher → even more impressions. Small advertisers
are disadvantaged.

```
Mitigations:
├─ Exploration budget (covered in cold start) helps new/small ads
├─ Quality score normalization: evaluate quality RELATIVE to
│  impression volume (use smoothed CTR, not raw click counts)
├─ Minimum impression guarantees for new campaigns in their first
│  week (burn revenue short-term, grow advertiser base long-term)
└─ Regular audits: are top-10 advertisers consuming >X% of
   impressions? If so, diversify.
```

### Click Fraud Detection

A real production concern: competitors or bots clicking on ads
to drain an advertiser's budget. This DIRECTLY hurts advertiser
ROI — our primary objective.

```
┌──────────────────────────────────────────────────────────────┐
│                CLICK FRAUD SIGNALS                            │
│                                                              │
│  Per-click signals:                                          │
│  ├─ IP address reputation (known bot/datacenter IPs)         │
│  ├─ Click-to-conversion time (instant = suspicious)          │
│  ├─ Mouse movement / touch patterns (if available)           │
│  ├─ Session behavior (clicked 50 ads in 10 seconds?)         │
│  └─ User agent anomalies (headless browser signatures)       │
│                                                              │
│  Aggregate signals:                                          │
│  ├─ Sudden CTR spike on a specific ad/keyword                │
│  ├─ Click clusters from same IP range / geo                  │
│  ├─ Abnormal click-through without any conversions           │
│  └─ Time-of-day patterns (all clicks at 3am?)                │
│                                                              │
│  Response:                                                   │
│  ├─ Real-time: flag suspicious clicks, don't charge advertiser│
│  ├─ Batch: retroactive refunds after daily fraud analysis     │
│  └─ Prevention: CAPTCHA for suspicious sessions, rate limiting│
│                                                              │
│  Model: anomaly detection (Isolation Forest or autoencoder)   │
│  trained on known-fraud examples + heuristic rules.           │
│  Runs ASYNCHRONOUSLY — doesn't add to serving latency.        │
└──────────────────────────────────────────────────────────────┘
```

### Delayed Conversions & Attribution

Conversions often happen hours or days after a click — user clicks
an ad for running shoes, thinks about it, buys next day.

```
Problem timeline:
─────────────────────────────────────────────────────
Day 1 12:00  User clicks ad           ← we log this
Day 1 12:01  User browses, leaves     ← no conversion yet
Day 2 09:00  User returns, purchases  ← conversion arrives
Day 3        Model trains on Day 1 data
             → this click has NO conversion label yet!
             → model learns "this was a non-converting click" ← WRONG
─────────────────────────────────────────────────────

Solutions:
├─ Attribution window: wait 7 days before using data for training
│  (loses freshness — trade-off)
├─ Label correction: train on recent data with partial labels,
│  then RE-TRAIN when conversions arrive and correct the labels
├─ Importance weighting: weight recent (uncertain) examples lower
│  than older (label-complete) examples
└─ Hybrid: use 7-day-old data as primary training set,
   supplement with last-24h data (click labels only, no conversion)
   to maintain freshness for CTR prediction
```

---

## Privacy & Industry Trends

```
Key constraints to design for:
├─ Cookie deprecation (3P cookies going away)
│  → rely more on first-party signals: query, on-site behavior
│  → contextual targeting gains importance over user targeting
│  → our search ads system is well-positioned: query intent is
│    first-party and doesn't need cookies!
│
├─ GDPR / CCPA compliance
│  → user data deletion: must be able to remove user_id embeddings
│    and history on request → design embedding tables to support
│    per-user deletion without full retraining
│  → consent management: only use engagement history if user opts in
│  → audit trail: log which features were used per prediction
│
└─ On-device signals (emerging)
   → privacy-preserving user features computed on-device,
     sent as opaque embeddings (like Google's Privacy Sandbox)
```

---

## Phased Rollout Plan

How I'd actually build this, in order:

```
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1 — MVP (Weeks 1-6)                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ • Keyword inverted index (exact + phrase match only)    │  │
│  │ • Logistic Regression P(click) model                    │  │
│  │ • Hand-crafted features: BM25, match type, ad hist CTR  │  │
│  │ • Simple GSP auction: bid × P(click)                    │  │
│  │ • Basic budget pacing (daily cap / 24 hours)            │  │
│  │ • Rule-based quality heuristic                          │  │
│  │                                                        │  │
│  │ Goal: END-TO-END pipeline working. Establish baseline   │  │
│  │ metrics. Start collecting training data at scale.       │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  PHASE 2 — MODEL UPGRADE (Weeks 7-14)                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ • DNN ranking model (DCN-v2) with multi-task heads      │  │
│  │ • Embedding tables for sparse features                  │  │
│  │ • Pre-computed BERT embeddings for query/ad text         │  │
│  │ • Online feature store (Redis) for real-time features   │  │
│  │ • Position bias correction in training                  │  │
│  │ • Post-hoc calibration (isotonic regression)            │  │
│  │                                                        │  │
│  │ Goal: +2-5% AUC improvement, better calibration,       │  │
│  │ A/B tested against LR baseline.                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  PHASE 3 — RETRIEVAL & COLD START (Weeks 15-22)              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ • Broad match via embedding retrieval (FAISS)           │  │
│  │ • Exploration budget for new ads                        │  │
│  │ • Feature fallback hierarchy                            │  │
│  │ • Content-based embedding prediction for new ads        │  │
│  │ • Click fraud detection (async pipeline)                │  │
│  │                                                        │  │
│  │ Goal: expand coverage (more ads eligible per query),    │  │
│  │ improve new-advertiser experience, protect ROI.         │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  PHASE 4 — OPTIMIZATION (Ongoing)                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ • Conversion prediction head (enough data by now)       │  │
│  │ • Thompson sampling for exploration                     │  │
│  │ • Real-time model updates (online learning)             │  │
│  │ • Learned budget pacing (RL-based)                      │  │
│  │ • Query intent classification for ad load decisions     │  │
│  │ • Privacy-preserving features                           │  │
│  │                                                        │  │
│  │ Goal: continuous improvement, handle edge cases,        │  │
│  │ optimize for long-term advertiser + user value.         │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## What Changes for Social/Feed Ads (Instagram)?

The fundamental shift: **no query → no explicit intent → everything pivots to user profiling and context.**

```
                    SEARCH ADS              SOCIAL FEED ADS
                    ──────────              ───────────────
Intent signal       Query (explicit)        Behavior history (implicit)
Retrieval           Keyword index           User-profile → ad targeting
                                            (interest, lookalike, retargeting)
Inventory           ~100K keyword-matched   ~Millions of ads eligible
                    candidates              for any user → retrieval is
                                            MUCH harder, needs ANN
Ranking signal      Query-ad relevance      User-ad affinity (predicted
                    dominates               from past behavior)
User features       Secondary (query is     PRIMARY — demographics,
                    king)                   interests, social graph,
                                            engagement history
Ad format           Text-heavy (title +     Visual-heavy (images, video,
                    description + URL)      Stories, Reels) → need
                                            visual understanding models
Cold start (user)   Low impact (query       HIGH impact — new users have
                    carries intent)         no profile → can't target
Latency budget      ~50ms (user is          ~200ms (user is scrolling,
                    actively waiting)       more tolerant)
Ad load decision    Show ads on every       How many ads per N organic
                    commercial query        posts? (ad load ratio is a
                                            critical UX lever)
Auction trigger     Every query             Every feed refresh / scroll
```

**Key architectural changes:**

1. **Retrieval** becomes ML-heavy — no keywords to match on.
   Use user embedding → ANN against ad embeddings (FAISS).
   Also: audience targeting (advertiser defines "women 25-34
   interested in fitness") → inverted index on user segments.

2. **User modeling** becomes the core ML problem — need a rich
   user representation from engagement history, social graph,
   content interactions. Sequence models (Transformers on user
   action history) replace query features as the primary signal.

3. **Visual understanding** matters — ad creative quality (image
   aesthetics, video engagement prediction) becomes a ranking
   feature. Need a CNN/ViT sub-model for creative scoring.

4. **Negative feedback** is more important — users can't "not search,"
   but they CAN hide/report feed ads. Predicting P(hide) or
   P(negative feedback) becomes an explicit model head to protect UX.

5. **Multi-stage retrieval** is essential — can't score millions of ads
   per request. Need: coarse retrieval (ANN, ~10K) → pre-ranking
   (lightweight model, ~1K) → full ranking (heavy model, ~100).
   Search ads often skip the pre-ranking stage.

---

## Key Trade-offs & Decisions Summary

| Decision                    | Choice              | Reason                                    | Would revisit if...               |
|-----------------------------|---------------------|-------------------------------------------|-----------------------------------|
| Retrieval                   | Keyword inverted idx| Natural fit for search ads, fast           | Inventory grows to billions       |
| Ranking model               | DCN-v2 / DLRM      | Handles sparse features, fast inference    | Need deep text understanding      |
| Multi-task vs separate      | Multi-task          | Shared representation, less infra          | Task interference is high         |
| Auction                     | GSP                 | Industry standard, incentive compatible    | Move to VCG for truthfulness      |
| Training frequency          | Daily + RT features | Balance freshness vs cost                  | Concept drift is faster           |
| Calibration method          | Isotonic regression | Non-parametric, handles any shape          | Scale requires simpler method     |