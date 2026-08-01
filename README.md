# MSCS 634 - Lab Assignment 5

**Name:** Chun-Shuo Sheu
**Course:** MSCS 634 - Big Data & Data Mining
**Lab:** Lab 5 - Clustering Analysis with Hierarchical and DBSCAN Clustering

## Purpose

The purpose of this lab is to explore two clustering families that define "cluster" in fundamentally
different ways, using the Wine Dataset from `sklearn`:

- **Agglomerative Hierarchical Clustering** - merges the closest pair of clusters bottom-up until a
  single tree remains, so `n_clusters` is chosen *after* fitting by cutting the dendrogram.
- **DBSCAN** - treats clusters as dense regions separated by sparse ones, discovers the cluster count
  on its own, and explicitly labels low-density points as **noise**.

Every clustering is evaluated with the three metrics required by the assignment:

- **Silhouette Score** - internal measure of compactness and separation (uses no labels).
- **Homogeneity Score** - external: does each cluster contain only one true class?
- **Completeness Score** - external: is each true class kept together in one cluster?

The **Adjusted Rand Index (ARI)** is reported as well, as a single-number summary of agreement with
the ground-truth cultivars. A 2D PCA projection (55.4% of variance) is used for all visualizations,
while all clustering and all metrics are computed in the full 13-dimensional standardized space.

## How to Reproduce

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib scikit-learn scipy jupyter
jupyter notebook MSCS_634_Lab_5.ipynb
```

## Results

The dataset is 178 samples x 13 features, 3 cultivars (59 / 71 / 48), with no missing values and no
duplicate rows.

### Hierarchical clustering - linkage x n_clusters

| Linkage | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|
| **ward** (Silhouette / ARI) | 0.267 / 0.439 | **0.277 / 0.790** | 0.226 / 0.657 | 0.187 / 0.583 | 0.180 / 0.576 |
| complete | 0.162 / 0.295 | 0.204 / 0.577 | 0.194 / 0.643 | 0.188 / 0.678 | 0.176 / 0.663 |
| average | 0.259 / -0.002 | 0.158 / -0.005 | 0.149 / -0.011 | 0.230 / 0.431 | 0.215 / 0.425 |
| single | 0.223 / -0.005 | 0.183 / -0.007 | 0.181 / -0.008 | 0.142 / -0.010 | 0.121 / -0.011 |

Ward with `n_clusters=3` is the best configuration in the entire sweep on both silhouette and ARI.

### DBSCAN parameter grid (excerpt)

| eps | min_samples | Clusters | Noise | Silhouette | Homogeneity | Completeness | ARI |
|---|---|---|---|---|---|---|---|
| 1.50 | 5 | 0 | 178 (100%) | n/a | 0.000 | 1.000 | 0.000 |
| 1.75 | 5 | 6 | 132 (74%) | -0.208 | 0.263 | 0.283 | 0.039 |
| 2.20 | 5 | 2 | 55 (31%) | 0.143 | 0.410 | 0.428 | 0.329 |
| 2.40 | 5 | 2 | 36 (20%) | 0.196 | 0.504 | 0.552 | 0.421 |
| 2.50 | 10 | 2 | 32 (18%) | 0.204 | 0.505 | 0.560 | 0.424 |
| **2.75** | **15** | **2** | **25 (14%)** | **0.225** | **0.498** | **0.581** | **0.423** |
| 3.00 | 5 | 1 | 11 (6%) | n/a | 0.031 | 0.145 | -0.006 |

### Final comparison

| Metric | Hierarchical (Ward, k=3) | DBSCAN (eps=2.75, min_samples=15) |
|---|---|---|
| Silhouette | **0.277** | 0.225 |
| Homogeneity | **0.790** | 0.498 |
| Completeness | **0.783** | 0.581 |
| ARI | **0.790** | 0.423 |
| Clusters found | 3 (specified) | 2 (discovered) |
| Noise points | 0 | 25 (14%) |

Hierarchical recovers the three cultivars almost exactly - cluster sizes 58 / 56 / 64 against true
counts 59 / 71 / 48, with only **13 of 178** samples misassigned, all of them from `class_1`. DBSCAN
never separates `class_0` from `class_1`: its large cluster holds 58 `class_0` plus 49 `class_1`, it
isolates `class_2` cleanly (44 of 48), and it discards 25 points as noise, 20 of which are `class_1`.

## Key Insights

- **Hierarchical clustering (Ward, k=3) wins on every metric** - ARI 0.790 vs. 0.423, homogeneity
  0.790 vs. 0.498, completeness 0.783 vs. 0.581, and even the label-free silhouette (0.277 vs.
  0.225). It recovers all three cultivars; DBSCAN recovers two.
- **DBSCAN's failure is structural, not a tuning problem.** DBSCAN can only split two groups when a
  genuine *density valley* separates them, and in the standardized 13-D space there is no valley
  between cultivars 0 and 1 - they form one continuous dense region. Ward does not need a valley; it
  only needs within-group variance to be smaller than between-group variance, which holds here.
- **The k-distance plot predicted this in advance.** The sorted k-th-nearest-neighbor curves rise
  smoothly with no elbow (median 1.9-2.8 depending on k), meaning there is no natural density
  threshold to find. A missing elbow is a warning that DBSCAN is the wrong tool, and it was.
- **The dendrogram found the cluster count on its own.** The last two merges occur at heights 35.4
  and 27.7 while the next merge down sits at 12.6 - a factor-of-two gap that unambiguously indicates
  three groups. Below height ~12 the merge heights become dense and gradual, i.e. further splits only
  carve up within-group variation.
- **Homogeneity and completeness must be read as a pair.** Past k=3, Ward's homogeneity kept rising
  (0.790 → 0.835 at k=6) while completeness collapsed (0.783 → 0.553). Splitting a true class into
  two pure fragments never hurts homogeneity, so optimizing it alone rewards over-splitting - a
  clustering with one point per cluster scores a perfect 1.0.
- **Linkage choice mattered more than `n_clusters`.** At k=3, Ward reached ARI = 0.790, `complete`
  only 0.577, and `average` and `single` collapsed to ARI ≈ 0.0 (one giant cluster plus singletons).
  Single linkage chains through the continuum between cultivars 0 and 1 for exactly the same reason
  DBSCAN does - both propagate connectivity through nearest neighbors.
- **`eps` had a very narrow working window.** At `eps ≤ 1.25` every point was noise regardless of
  `min_samples`; at 1.75 the result was 6 fragments and 132 noise points; 2.2-2.75 gave two sensible
  clusters; by 3.0 everything collapsed into a single cluster. The entire useful range spans about
  0.5 units of distance.
- **`min_samples` pushes the same trade-off the other way.** At fixed `eps = 2.4`, raising
  `min_samples` from 3 to 15 grew the noise set from 28 to 80 points and lowered both external
  metrics, because core points become harder to qualify and clusters fragment.
- **Silhouette alone nearly picked a degenerate solution.** The runner-up configuration
  (`eps=3.5, min_samples=3`) scored a silhouette of 0.216 against the winner's 0.225 - yet it is 172
  points in one cluster, 3 in another, with homogeneity 0.03. Silhouette is perfectly happy with "one
  blob plus a speck". The external metrics are what exposed it.
- **The two DBSCAN metric failures are different failures.** Low homogeneity (0.498) means its
  clusters *mix* classes; low completeness (0.581) means one class is *scattered* across a cluster
  and the noise set. Both were happening at once, and neither metric alone would have shown it.
- **The result says more about the data than the algorithms.** The Wine cultivars are compact,
  roughly convex, contiguous, and free of meaningful outliers - the ideal setting for Ward and the
  worst-case setting for DBSCAN. On data with irregular cluster shapes, real outliers, or an unknown
  number of well-separated dense regions, the ranking would plausibly reverse.

## Strengths and Weaknesses Observed

**Hierarchical clustering.** *Strengths:* the dendrogram is a genuine analysis tool - it revealed the
correct cluster count from the data rather than requiring a guess, and it exposes structure at every
scale at once. Ward handled the touching, elongated cultivar groups without difficulty, assigned
every point, and is deterministic (no random initialization). *Weaknesses:* it forces every point
into a cluster, so a real outlier necessarily contaminates one; merges are greedy and irreversible;
it needs O(n²) memory for the distance matrix, which is fine at n=178 but rules it out at scale; and
results swing wildly with the linkage rule (ARI 0.79 down to 0.0 across the four tested).

**DBSCAN.** *Strengths:* it discovered the cluster count itself and separated 25 low-density points
as noise instead of forcing them into a cluster - and those noise labels are informative, since the
20 rejected `class_1` samples really do sit in the sparse transition zone between the other two
cultivars. It is the right tool for arbitrarily shaped clusters and data with true outliers.
*Weaknesses:* acute sensitivity to `eps`, a 2D parameter search with interacting parameters, a single
global `eps` that cannot adapt to clusters of differing density, and a hard dependence on density
gaps that simply do not exist in this dataset. In 13 dimensions the concentration of pairwise
distances further erodes the density contrast it needs.

## Challenges and Decisions

- **Standardization was non-negotiable.** Raw feature standard deviations span four orders of
  magnitude (`proline` at 314.9 vs. `nonflavanoid_phenols` at 0.12). Since Ward distances and the
  DBSCAN `eps` radius are both Euclidean, `proline` alone would have dictated every distance. I
  applied z-score normalization and verified the resulting means (≈0) and standard deviations (≈1).
- **Choosing `eps` honestly instead of by trial and error.** I generated k-distance plots for
  `min_samples` ∈ {3, 5, 10, 15} first to bracket the plausible range, then ran a 9 x 5 grid over
  `eps` x `min_samples`. The absence of an elbow in those curves is itself a reported finding, not a
  detail to skip past.
- **Silhouette is ambiguous when noise exists.** Computing it with noise treated as its own label
  penalizes runs that dump most points into noise; computing it on the clustered points only rewards
  them (`eps=2.2, min_samples=15` scores 0.540 on that basis while discarding 66% of the data). I
  report **both** columns and select on the conservative all-points version, with the `n_clusters ≥ 2`
  constraint noted below.
- **Guarding the model-selection step.** Silhouette is undefined for a single group, and a
  one-cluster-plus-noise run can score deceptively well. Selection was restricted to runs finding at
  least 2 clusters, and I documented the near-degenerate runner-up explicitly rather than letting the
  ranking imply that the choice was clear-cut.
- **Sweeping linkage, not just `n_clusters`.** The assignment asks for different `n_clusters` values,
  but testing only Ward would have hidden the more important result - that `average` and `single`
  linkage fail outright on this data. Sweeping both dimensions is what made the comparison with
  DBSCAN's connectivity-chaining failure possible.
- **Dendrogram readability.** A 178-leaf dendrogram is unreadable at the leaves, so the notebook
  plots the full tree (with the 2- and 3-cluster cut heights marked) alongside a `truncate_mode='lastp'`
  view of the last 12 merges with contracted cluster sizes shown.
- **Visualization vs. computation space.** PCA to 2D is used *only* for plotting. Two components
  capture 55.4% of the variance, so the plots understate the true separation - all clustering and all
  metrics are computed on the full 13 standardized features.
- **Reproducibility.** Both algorithms here are deterministic (no random initialization), and
  `random_state=42` is fixed for PCA, so results are identical run-to-run.

## Files

| File | Contents |
|---|---|
| `MSCS_634_Lab_5.ipynb` | The complete lab notebook with all output |
| `hierarchical_metrics.png` | Metric response to `n_clusters` under Ward linkage |
| `hierarchical_clusters.png` | Ward clusters for k=2...6 plus ground truth (PCA projection) |
| `dendrogram.png` | Full and truncated Ward dendrograms with cut heights |
| `k_distance_plot.png` | k-distance curves used to bracket `eps` |
| `dbscan_clusters.png` | Six `eps`/`min_samples` combinations, noise highlighted |
| `cluster_comparison.png` | Ground truth vs. Hierarchical vs. DBSCAN, side by side |
| `metric_comparison.png` | Bar chart of all four metrics for both final models |
| `requirements.txt` | Pinned environment (`pip freeze`) |
