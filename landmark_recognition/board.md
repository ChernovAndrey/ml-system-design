# Landmark Recognition System — ML System Design

---

## 1. Requirements Summary

| Dimension        | Constraint                                      |
|------------------|--------------------------------------------------|
| Scale            | Arbitrary # of landmarks, dynamically growable   |
| Input            | Image (primary), GPS/metadata (optional)         |
| Latency          | ≤ 2–3 sec per prediction                         |
| Precision/Recall | Precision-first; return "unknown" when unsure    |
| Adding landmarks | Without full model retraining                    |

---

## 2. High-Level Architecture

```
                         ┌──────────────┐
                         │  User Image  │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  Preprocessing│  (resize, validate, orientation)
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  Landmark     │  "Is there a landmark at all?"
                         │  Detector     │  (binary filter — fast, high recall)
                         └──────┬───────┘
                                │ yes
                         ┌──────▼───────┐
                         │  Embedding    │  CNN/ViT → d-dim vector
                         │  Model        │
                         └──────┬───────┘
                                │
                    ┌───────────▼────────────┐
                    │  Retrieval (ANN Index)  │  query top-K from landmark DB
                    │  (FAISS / ScaNN / etc.) │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Re-ranking / Scoring   │  geometric verify + confidence
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Confidence Gate        │  score ≥ τ → label
                    │                        │  score < τ → "unknown"
                    └───────────┬────────────┘
                                │
                         ┌──────▼───────┐
                         │   Response    │  landmark name, metadata, confidence
                         └──────────────┘
```

---

## 3. Metrics

### Offline
- **Precision@1** (primary) — when we return a label, is it correct?
- **Recall@K** — is the correct landmark in the top-K retrieved?
- **"Unknown" rate** — what fraction do we decline? (monitor, not optimize)
- **GAP (Global Average Precision)** — standard metric from Google Landmark challenges

### Online
- User engagement (clicks, shares, saves after recognition)
- Correction rate (user overrides our label)
- Latency p50 / p95

---

## 4. Baseline → V1 → V2 Progression

### Baseline (v0) — No training at all
- Take off-the-shelf ImageNet-pretrained ResNet-50 / EfficientNet
- Extract penultimate-layer features (2048-d) as embeddings
- Build a brute-force KNN index over reference images
- Cosine similarity → threshold → label or "unknown"
- **Why start here**: tells us how much signal we get "for free"
  from general visual features. Sets the floor.

### V1 — Fine-tuned embedding model
- Fine-tune backbone with metric learning (see §5)
- Replace brute-force with ANN index (FAISS IVF-PQ)
- Add confidence gating
- **Expected lift**: large — metric learning learns to pull
  same-landmark images together and push different ones apart

### V2 — Full pipeline
- Add landmark detector stage
- Geometric verification in re-ranking
- Optional GPS prior when available
- Query expansion / multi-scale features

---

## 5. Data

### Training Data
| Source                         | Size         | Notes                          |
|--------------------------------|--------------|--------------------------------|
| Google Landmarks v2 (GLDv2)    | ~5M images   | ~200K landmarks, long-tail     |
| Web-scraped (Flickr, Wiki)     | variable     | needs cleaning, noisy labels   |
| User-contributed (if product)  | grows w/time | implicit labels via GPS + maps |

### Landmark Reference Database
- For each landmark: curated set of **reference images** (5–50+)
  - Multiple angles, lighting, seasons
  - Each reference image → pre-computed embedding vector
- Metadata: name, GPS coords, category, description, Wiki link

### Data Challenges
- **Long tail**: Eiffel Tower has millions of photos; a local
  church might have 5. Need to handle class imbalance.
- **Noise**: web-scraped images often mislabeled, contain
  interiors, souvenirs, etc.
- **Near-duplicates**: same landmark, extremely similar viewpoint
  → can inflate metrics. Need de-duplication.

### Data Cleaning Pipeline
```
Raw images → De-dup (perceptual hash) → Face/text filter
  → Manual/crowdsource label verification (for reference set)
  → Train/val/test split (by landmark, NOT random)
```
**Important**: split by landmark for test, so we evaluate
generalization to landmarks not seen in training (open-set).

---

## 6. Model Design

### Embedding Model (core component)

```
┌───────────────────────────────────────────────────┐
│                                                   │
│   Input Image (224×224 or 512×512)                │
│         │                                         │
│   ┌─────▼──────┐                                  │
│   │  Backbone   │  EfficientNet-B4 / ViT-B/16    │
│   │ (pretrained)│  (ImageNet or CLIP pretrained)  │
│   └─────┬──────┘                                  │
│         │                                         │
│   ┌─────▼──────┐                                  │
│   │  GeM Pool   │  Generalized Mean Pooling       │
│   │             │  (better than avg pool for       │
│   │             │   retrieval tasks)               │
│   └─────┬──────┘                                  │
│         │                                         │
│   ┌─────▼──────┐                                  │
│   │  FC → 512-d │  Projection head                │
│   │  L2-norm    │  Unit hypersphere               │
│   └─────┬──────┘                                  │
│         │                                         │
│     embedding (512-d, L2-normalized)              │
│                                                   │
└───────────────────────────────────────────────────┘
```

### Why these choices?
- **GeM pooling** > global avg pooling: learnable pooling that
  emphasizes discriminative regions. Proven in landmark retrieval
  (Radenovic et al.).
- **512-d**: good balance of expressiveness vs. index size.
  At 1M landmarks × 50 refs = 50M vectors × 512 × 4B ≈ 100 GB
  → manageable with PQ compression.
- **L2-norm**: maps to unit sphere → cosine sim = dot product
  → fast, and works well with ANN libraries.

### Training: Metric Learning

**Loss function — ArcFace (Additive Angular Margin)**

- Treats training as classification with angular margin penalty
- Produces highly discriminative embeddings
- Better than triplet loss (no hard mining needed) and
  contrastive loss (more stable training)

```
L = -log( e^(s·cos(θ_y + m)) / (e^(s·cos(θ_y + m)) + Σ e^(s·cos(θ_j))) )

s = scale (e.g., 30)
m = angular margin (e.g., 0.5)
```

**Trade-off note**: I chose ArcFace over triplet loss because:
- Triplet loss requires careful hard-negative mining
- ArcFace uses all classes in the batch implicitly
- ArcFace has shown superior performance on retrieval benchmarks
- Caveat: needs a classification head during training (one class
  per landmark), but we discard it at inference — only keep embeddings

### Training Details
- Optimizer: AdamW, cosine LR schedule, warmup
- Augmentation: random crop, color jitter, rotation, cutout
  (but NOT flips — landmarks are not symmetric in general,
   though horizontal flip is usually fine)
- Input resolution: train at 224, fine-tune at 512 (progressive resizing)
- Batch size: large (512+) — more negatives per batch helps

---

## 7. Retrieval System

### Landmark Index

```
┌─────────────────────────────────────────────┐
│            Landmark Database                 │
│                                             │
│  Landmark_ID │ Name         │ Refs (embeds) │
│  ────────────┼──────────────┼───────────────│
│  L001        │ Eiffel Tower │ [v1,v2,...v50]│
│  L002        │ Taj Mahal    │ [v1,v2,...v30]│
│  ...         │ ...          │ ...           │
│  L_N         │ Local church │ [v1,v2,v3]   │
│                                             │
│  Total: ~50M vectors (1M landmarks × 50)   │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         FAISS Index (IVF + PQ)              │
│                                             │
│  IVF: partition into ~10K clusters          │
│  PQ:  compress 512-d → 64 bytes per vector │
│  → total index size: ~3.2 GB (fits in RAM) │
│                                             │
│  Query: probe top-64 clusters → return K=100│
└─────────────────────────────────────────────┘
```

### Query Flow (detailed)
1. Query image → embedding model → 512-d vector
2. ANN search → top-K=100 nearest reference vectors
3. **Aggregate by landmark**: group results by landmark_id,
   compute per-landmark score (e.g., avg similarity of top-5
   matches for that landmark)
4. **Re-rank**: optional geometric verification (RANSAC on
   local features) for top-3 candidates
5. **Confidence gate**: top score ≥ τ → return label;
   else → "unknown"

### Re-ranking Pipeline (detailed)

ANN gives us top-K=100 raw results sorted by global embedding
cosine similarity. That's a rough filter. Re-ranking refines it:

```
ANN top-100 results (reference vectors with landmark_ids)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Step 1: Aggregate by landmark                    │
│                                                   │
│  Group 100 results by landmark_id.                │
│  Per landmark score = mean similarity of its      │
│  top-5 matched reference images.                  │
│                                                   │
│  Why top-5 mean, not max?                         │
│  Max is noisy (one lucky match). Mean over top-5  │
│  requires consistent similarity across multiple   │
│  viewpoints → much more robust.                   │
│                                                   │
│  Output: ranked list of ~20-40 candidate landmarks│
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│  Step 2: Geometric Verification (top-3 candidates)│
│                                                   │
│  For each of the top-3 landmark candidates:       │
│                                                   │
│  Query image    Matched reference                 │
│  ┌──────────┐   ┌──────────┐                      │
│  │ ● ●      │   │  ● ●     │  Extract local      │
│  │   ●  ●   │   │    ● ●   │  keypoints +        │
│  │ ●     ●  │   │  ●    ●  │  descriptors        │
│  └──────────┘   └──────────┘  (DELF or SuperPoint)│
│       │               │                           │
│       └───────┬───────┘                           │
│               ▼                                   │
│    Match local descriptors (mutual nearest        │
│    neighbor matching)                             │
│               │                                   │
│               ▼                                   │
│    RANSAC → estimate affine/homography transform  │
│               │                                   │
│               ▼                                   │
│    Count inliers (spatially consistent matches)   │
│                                                   │
│  Inliers ≥ 15 → ✅ geometric match (strong)      │
│  Inliers < 15 → ❌ likely false positive          │
│                                                   │
│  This catches the key failure mode:               │
│  Two Gothic churches may have similar global      │
│  embeddings, but their detailed spatial layout    │
│  of windows, doors, ornaments differs →           │
│  RANSAC will fail → we don't confuse them.        │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│  Step 3: Final scoring                            │
│                                                   │
│  final_score = α · global_sim                     │
│              + β · geo_verify_score               │
│              + γ · match_count_norm               │
│              + δ · gps_prior (if available)        │
│                                                   │
│  Where:                                           │
│  - global_sim: aggregated embedding similarity    │
│  - geo_verify_score: inlier_count / total_matches │
│  - match_count_norm: how many of this landmark's  │
│    reference images appeared in top-100            │
│  - gps_prior: exp(-distance / σ) if GPS available │
│    (Gaussian decay, σ ≈ 10km)                     │
│                                                   │
│  α, β, γ, δ tuned on validation set.              │
│  (Could also learn these weights with a small     │
│   logistic regression on top — cheap and tunable) │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
              Confidence Gate (§8)
```

### DELF Architecture (DEep Local Features, Google 2017)
Open source: tensorflow/models/research/delf

```
Input Image
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  Backbone: ResNet-50 (pretrained, fine-tuned)        │
│                                                      │
│  Extract feature maps from conv4 layer               │
│  Output: H × W × C feature map                      │
│  (e.g., 28×28 spatial, 1024 channels)               │
│                                                      │
│  Each spatial location = one potential keypoint      │
│  with a 1024-d descriptor                            │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Attention Module (what makes DELF special)          │
│                                                      │
│  Small 2-layer FC network per spatial location:      │
│                                                      │
│  feature_i (1024-d) → FC(512) → ReLU → FC(1) →      │
│  softplus → attention_score_i                        │
│                                                      │
│  This learns WHICH regions are important for         │
│  landmark recognition:                               │
│    High attention: ornate doorway, unique spire,     │
│                    carved relief, distinctive window  │
│    Low attention:  sky, grass, people, generic wall   │
│                                                      │
│  Trained with landmark classification loss —         │
│  attention learns to focus on discriminative parts   │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Selection & Output                                  │
│                                                      │
│  1. Rank all H×W locations by attention score        │
│  2. Keep top-K (e.g., K=500) keypoints               │
│  3. Record each keypoint's:                          │
│     - (x, y) position in the image                   │
│     - descriptor vector (1024-d, PCA → 40-d)         │
│     - attention score (used as confidence)            │
│                                                      │
│  PCA 1024→40 reduces storage & speeds matching       │
│  while retaining >95% discriminative power           │
│                                                      │
│  Output per image: ~500 keypoints, each with         │
│  (x, y, 40-d descriptor, score)                      │
└──────────────────────────────────────────────────────┘
```

### RANSAC Geometric Verification Pipeline

Once we have local features from both images, here's how
geometric verification works:

```
Query image keypoints          Reference image keypoints
(~500 keypoints, 40-d each)    (~500 keypoints, 40-d each)
         │                              │
         └──────────┬───────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│  Step A: Local Descriptor Matching                   │
│                                                      │
│  For each query keypoint, find nearest neighbor      │
│  in reference keypoints (L2 distance on 40-d).       │
│                                                      │
│  Apply Lowe's ratio test:                            │
│    dist(1st nearest) / dist(2nd nearest) < 0.7       │
│    → keep match (it's distinctive)                   │
│    → else reject (ambiguous match)                   │
│                                                      │
│  Output: ~50-200 tentative matches                   │
│  Each match: (x_q, y_q) ↔ (x_r, y_r)               │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Step B: RANSAC (Random Sample Consensus)            │
│                                                      │
│  Goal: find a geometric transform that maps query    │
│  keypoint positions to reference keypoint positions.  │
│                                                      │
│  for iteration in range(1000):                       │
│    1. Randomly pick 3 matches (minimum for affine)   │
│    2. Estimate affine transform T from these 3       │
│                                                      │
│       [x_r]   [a b tx] [x_q]                        │
│       [y_r] = [c d ty] [y_q]                        │
│       [ 1 ]   [0 0  1] [ 1 ]                        │
│                                                      │
│    3. Apply T to ALL query keypoints                 │
│    4. Count inliers: matches where                   │
│       |T(x_q, y_q) - (x_r, y_r)| < ε (e.g., 20px) │
│    5. Keep transform with most inliers               │
│                                                      │
│  Output:                                             │
│  - inlier_count: how many matches are spatially      │
│    consistent                                        │
│  - inlier_ratio: inlier_count / total_matches        │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Step C: Interpret Results                           │
│                                                      │
│  inliers ≥ 15   → MATCH — same landmark, the        │
│                    spatial layout of details aligns   │
│  inliers < 15   → NO MATCH — maybe similar style    │
│                    but different physical structure   │
│                                                      │
│  WHY THIS WORKS for similar-looking landmarks:       │
│                                                      │
│  Cathedral A         Cathedral B                     │
│  ┌────────────┐      ┌────────────┐                  │
│  │ 🪟  ⛪  🪟 │      │ 🪟  ⛪  🪟 │  Similar style! │
│  │ 🚪    🪟  │      │ 🪟    🚪  │  But doors,      │
│  │   🪟  🪟  │      │ 🪟  🪟    │  windows are in  │
│  └────────────┘      └────────────┘  different places│
│                                                      │
│  Global embedding: similar (same style)              │
│  Local features match: yes (similar elements)        │
│  RANSAC: FAILS — no single transform maps A's        │
│  layout to B's layout → correctly rejects match     │
└──────────────────────────────────────────────────────┘
```

### Alternatives to DELF (all open source)

| Model       | Source     | Pros                        | Cons                    |
|-------------|------------|-----------------------------|-------------------------|
| **DELF**    | Google     | Built for landmarks,        | TF-only, older (2017)   |
|             |            | attention module             |                         |
| **SuperPoint**| MagicLeap| Robust to viewpoint change, | Not landmark-specific   |
|             |            | real-time speed              |                         |
| **SuperGlue**| MagicLeap | Learned matcher (replaces   | Heavier, needs GPU for  |
|             |            | ratio test + RANSAC)        | matching step           |
| **DISK**    | EPFL       | Differentiable keypoints,   | Less battle-tested      |
|             |            | state-of-the-art matching   |                         |

**My choice**: DELF for V1 (domain match outweighs age).
Consider SuperPoint + SuperGlue for V2 if we want to replace
the hand-crafted RANSAC with a learned matching pipeline —
SuperGlue jointly solves matching + filtering, often better
than ratio test + RANSAC, but at higher compute cost.

**Cost note**: geometric verification is expensive (~50-70ms per
pair), which is why we only run it on the top-3 candidates, not
all 100 ANN results. Within our latency budget.

### Adding a New Landmark (zero retraining)
1. Collect 5–50 reference images of the new landmark
2. Run each through embedding model → get vectors
3. Insert vectors + metadata into FAISS index
4. Done! The landmark is now recognizable.

This is the key advantage of retrieval over classification.

---

## 8. Confidence & "Unknown" Handling

Since precision >> recall, we need a robust confidence gate:

**Score = similarity of best-matching landmark**

Decision rule:
- score ≥ τ_high  → return label (high confidence)
- τ_low ≤ score < τ_high → return label + "uncertain" flag
- score < τ_low → return "unknown"

**Calibrating τ**: set on a held-out validation set to achieve
target precision (e.g., 95%). This is a precision-recall knob.

**Additional confidence signals**:
- Gap between #1 and #2 landmark scores (large gap = more confident)
- Number of reference hits for the top landmark
- Consistency of matched reference viewpoints

---

## 9. Landmark Detection & Cropping Strategy

### Option A: Explicit detector (crop landmark bbox)
```
Image → Object Detector → crop bbox → Embedding model
```
- ❌ "Landmark" is not a well-defined bbox class
- ❌ Adds ~200ms latency + model complexity
- ❌ Tight crop may lose useful context (river near Eiffel Tower)
- ✅ Cleanest input to the embedding model

### Option B: No cropping — rely on model attention (★ recommended)
```
Image → Embedding model (with attention) → embedding
```
- ✅ Simpler pipeline, no extra model
- ✅ GeM pooling + self-attention (ViT) naturally focus on
     discriminative regions
- ✅ Keeps useful surrounding context
- ✅ Multi-scale query handles varying landmark sizes
- ❌ Noisy images (small landmark, lots of clutter) may hurt

### Option C: Attention-guided cropping (V2 enhancement)
```
Image → Embedding model → attention map → re-crop → re-embed
         (pass 1: coarse)                  (pass 2: focused)
```
- ✅ Best of both worlds: first pass finds the region of
     interest, second pass gets a focused embedding
- ❌ 2× inference cost (~400ms, still within budget)
- Good candidate for V2 if Option B shows weakness on
  cluttered images

### Recommended approach:

**V1: Option B** — Use a ViT backbone (self-attention gives us
region focus for free) + GeM pooling + multi-scale extraction.

**How multi-scale works**:
```
Original image (full)     → embed → v_full
Center crop (70%)         → embed → v_center
Three 50% overlapping     → embed → v_crop1, v_crop2, v_crop3
crops

Final query = concatenate or aggregate these vectors
```
This implicitly handles the "where is the landmark?" problem —
one of these scales will capture the landmark well.

**V2: Option C** — if we see retrieval failures on cluttered
images, add an attention-guided second pass.

### The "is this a landmark at all?" filter

This is separate from cropping — it's a binary gate to avoid
wasting compute on non-landmark photos.

**Approach**: lightweight scene classifier
- Classes: {outdoor_architecture, monument, natural_landmark,
            bridge, statue, ...} vs {food, selfie, indoor, etc.}
- Model: MobileNet-v3 or small EfficientNet (~5ms inference)
- Threshold for high recall (>99%) — let borderline cases through
- This is NOT detecting where the landmark is, just whether
  the image is plausibly a landmark photo

```
Image → Scene Classifier → "landmark-like"?
              │                    │
              │ no                 │ yes
              ▼                    ▼
        Return "unknown"     Continue pipeline
        (fast path ~10ms)    (embedding + retrieval)
```

---

## 10. Serving & Scaling

### Serving Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              API Gateway                 │
                    │         (rate limit, auth, routing)      │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │          Orchestrator Service            │
                    │    (manages pipeline, timeout, fallback) │
                    └───┬──────────┬──────────────┬───────────┘
                        │          │              │
              ┌─────────▼──┐  ┌───▼──────┐  ┌───▼───────────┐
              │   Scene     │  │ Embedding│  │  Retrieval    │
              │ Classifier  │  │  Service │  │  Service      │
              │ (CPU, light)│  │ (GPU)    │  │  (FAISS, CPU) │
              └─────────────┘  └──────────┘  └───────────────┘
                                                    │
                                             ┌──────▼────────┐
                                             │ Landmark DB   │
                                             │ (metadata,    │
                                             │  vectors, etc)│
                                             └───────────────┘
```

### Latency Budget Breakdown (target: ≤ 2s p95)

| Stage               | Infra     | Latency    |
|----------------------|-----------|------------|
| Preprocessing        | CPU       | ~20ms      |
| Scene classifier     | CPU       | ~10ms      |
| Embedding (×3 scale) | GPU       | ~300ms     |
| ANN retrieval (K=100)| CPU/RAM   | ~10ms      |
| Aggregation + re-rank| CPU       | ~50ms      |
| Geometric verify (×3)| CPU       | ~200ms     |
| Network overhead     | —         | ~100ms     |
| **Total**            |           | **~700ms** |

Comfortably within the 2-3s budget. Leaves room for
Option C attention re-crop in V2 (~+300ms).

### Scaling Considerations

**Embedding service (GPU-bound)**
- Horizontal scale with GPU instances behind load balancer
- Batch requests where possible (group concurrent queries)
- Model optimization: TensorRT / ONNX for ~2× speedup
- Could use CPU inference for lower throughput (viable
  since latency budget is generous)

**Retrieval service (memory-bound)**
- FAISS index must fit in RAM (~3.2 GB with PQ)
- Replicate index across multiple nodes for throughput
- Sharding strategy if index grows beyond single-node RAM:
  - Shard by geographic region (if GPS available)
  - Or shard by IVF cluster assignment

**Index Updates (adding new landmarks)**
- New reference vectors → append to FAISS index
- FAISS supports add() without full rebuild
- Periodic re-train IVF centroids (e.g., weekly batch job)
  when index grows significantly (>20% since last train)
- Blue-green deployment: build new index in background,
  swap atomically

### Cost Optimization
- Scene classifier rejects ~60-70% of queries early → huge
  GPU savings (most photos aren't landmarks)
- GPU instances only needed for embedding service
- Cache: LRU cache on image hash → embedding (repeat queries)
- Could serve popular landmarks from a fast-path exact
  match cache (top 1000 landmarks cover most queries)

---

## 11. Failure Modes & Monitoring

### Failure Modes

| Failure                        | Impact           | Mitigation                         |
|--------------------------------|------------------|------------------------------------|
| Visually similar landmarks     | Wrong label      | Geometric verification + GPS prior |
| (e.g., many Gothic cathedrals) |                  | Multi-scale + re-rank gap check    |
| Landmark under construction    | Miss or wrong    | Periodic reference image refresh   |
| /scaffolding/seasonal change   |                  | Augmentation during training       |
| Unusual angle/night/fog        | Low confidence   | Diverse reference images; "unknown"|
|                                |                  | is acceptable (precision-first)    |
| Adversarial / trick images     | Wrong label      | Confidence gate catches most;      |
|                                |                  | geometric verify as second check   |
| Index stale after growth       | Degraded recall  | Monitor + periodic IVF retrain     |
| Embedding model drift          | Gradual quality  | Offline eval pipeline, retrain     |
|                                | degradation      | cycle (quarterly)                  |
| GPU service down               | Timeout          | Graceful fallback → "unknown"      |
|                                |                  | with retry; health checks          |

### Key edge case: Visually Similar Landmarks

This deserves extra attention. Many buildings share architectural
styles (Gothic churches, Greek temples, modern skyscrapers).

**Mitigation stack:**
1. Embedding model: ArcFace margin pushes similar-but-different
   landmarks apart in embedding space
2. Geometric verification: RANSAC on local features (DELF-style)
   verifies spatial consistency — same architectural style won't
   have identical spatial layout of details
3. GPS prior (when available): if user has GPS, filter candidates
   to landmarks within 50km radius → dramatically reduces
   confusion set
4. Confidence gap: if top-2 candidates are both "Gothic church"
   with similar scores → return "unknown" rather than guess wrong
   (precision-first policy)

### Monitoring & Observability

```
┌─────────────────────────────────────────────┐
│              Monitoring Dashboard            │
├─────────────────────────────────────────────┤
│                                             │
│  Real-time:                                 │
│  • Latency p50/p95/p99 per pipeline stage   │
│  • "Unknown" rate (spike = model issue?)    │
│  • Error rate / timeout rate                │
│  • Query volume & geographic distribution   │
│                                             │
│  Daily/Weekly:                              │
│  • Precision@1 on sampled + human-reviewed  │
│    queries (golden set evaluation)          │
│  • Embedding distribution drift (cosine     │
│    sim stats vs. baseline)                  │
│  • Index coverage (new landmarks requested  │
│    but not in DB — expansion signal)        │
│  • User correction rate (if UI supports)    │
│                                             │
│  Alerts:                                    │
│  • "Unknown" rate > 2σ from baseline        │
│  • p95 latency > 2s                         │
│  • Precision drop on daily golden set       │
│                                             │
└─────────────────────────────────────────────┘
```

### A/B Testing & Iteration

- **What to A/B test**: new model versions, threshold values τ,
  multi-scale vs. single-scale, GPS prior on/off
- **Primary metric**: Precision@1 (guardrail: "unknown" rate
  doesn't spike beyond acceptable level)
- **Segmentation**: split by landmark popularity, geographic
  region, image quality — improvements may vary across segments
- **Shadow mode**: new models run in parallel, log predictions
  but don't serve — compare offline before promoting

---

## 12. Query Expansion (retrieval boost)

A known technique in image retrieval to improve recall
without changing the model:

```
Query image → embedding v_q
                │
                ▼
        ANN search → top-K matches
                │
                ▼
        Take top-3 matched reference embeddings
        (the ones we're most confident about)
                │
                ▼
        v_expanded = α · v_q + (1-α) · mean(v_top3)
        (α ≈ 0.7 — still dominated by original query)
                │
                ▼
        Second ANN search with v_expanded
                │
                ▼
        Merged results → re-rank as before
```

**Why this helps**: the query photo might show the landmark
from an unusual angle. Blending with known good reference
embeddings "pulls" the query toward the right cluster,
surfacing matches that the original query missed.

**Cost**: one extra ANN query (~10ms). Very cheap for
meaningful recall improvement. Within latency budget.

---

## 13. Cold Start: New Landmarks with Few References

When a new landmark is added with only 3-5 reference images,
retrieval quality may be poor. Mitigations:

1. **Synthetic augmentation of references**: apply heavy
   augmentations (viewpoint, lighting, crops) to the few
   reference images → expand to ~50 pseudo-references.
   Not as good as real diversity, but fills the gap.

2. **Lower confidence threshold temporarily**: for landmarks
   with <10 references, require only global similarity
   (skip geometric verification — not enough reference
   variety for reliable spatial matching). Accept higher
   "unknown" rate until we accumulate more references.

3. **Active collection**: when query images match a
   cold-start landmark with moderate confidence, flag
   them for human review. If confirmed → add to reference
   set. The landmark bootstraps itself.

---

## 14. Data Flywheel (continuous improvement)

This is what separates a good system from a great one —
the product gets smarter as people use it:

```
┌─────────────────────────────────────────────────┐
│                DATA FLYWHEEL                     │
│                                                  │
│  Users upload photos                             │
│       │                                          │
│       ▼                                          │
│  System recognizes → user confirms/corrects      │
│       │                                          │
│       ▼                                          │
│  Confirmed images → new reference candidates     │
│       │                                          │
│       ▼                                          │
│  Human review (sample) → add to reference DB     │
│       │                                          │
│       ▼                                          │
│  Better reference coverage → better recognition  │
│       │                                          │
│       ▼                                          │
│  Better recognition → more users → more photos   │
│       └──────────────────────────────────────────┘
│                                                  │
│  Quarterly: retrain embedding model on expanded  │
│  dataset → improved embeddings → re-index all    │
│  reference vectors                               │
└─────────────────────────────────────────────────┘
```

**Key signals from user behavior**:
- User confirms label → positive training example
- User corrects label → hard negative (very valuable!)
- User searches same landmark repeatedly → coverage gap
- High "unknown" rate for a GPS cluster → missing landmark

**Privacy note**: user photos used for training/reference
only with explicit opt-in consent. GPS data anonymized
and aggregated. Reference DB uses only consented images.

---

## 15. Summary of Key Design Decisions

| Decision                    | Choice               | Reason                              |
|-----------------------------|----------------------|-------------------------------------|
| Classification vs retrieval | Retrieval            | Scalable, add landmarks w/o retrain |
| Explicit crop vs attention  | Attention + multi-scale | Simpler, keeps context           |
| Loss function               | ArcFace              | Stable, no hard-mining needed       |
| Backbone                    | ViT-B/16             | Built-in attention, strong features |
| Pooling                     | GeM                  | Proven for retrieval tasks          |
| Embedding dim               | 512-d, L2-normalized | Balance of quality vs index size    |
| ANN index                   | FAISS IVF-PQ         | Fast, compressed, fits in RAM       |
| Re-ranking                  | Aggregate + DELF + RANSAC | Catches visually similar landmarks |
| Confidence strategy         | Dual threshold + gap | Precision-first with "unknown"      |
| GPS usage                   | Optional Gaussian prior | Huge help when available, not required |
| Local features              | DELF (V1), SuperGlue (V2) | Domain-specific → general      |
| Iteration                   | Data flywheel        | System improves with usage          |

| Decision                    | Choice               | Reason                              |
|-----------------------------|----------------------|-------------------------------------|
| Classification vs retrieval | Retrieval            | Scalable, add landmarks w/o retrain |
| Explicit crop vs attention  | Attention + multi-scale | Simpler, keeps context           |
| Loss function               | ArcFace              | Stable, no hard-mining needed       |
| Backbone                    | ViT-B/16             | Built-in attention, strong features |
| Pooling                     | GeM                  | Proven for retrieval tasks          |
| ANN index                   | FAISS IVF-PQ         | Fast, compressed, fits in RAM       |
| Confidence strategy         | Dual threshold + gap | Precision-first with "unknown"      |
| GPS usage                   | Optional prior       | Huge help when available, not required|