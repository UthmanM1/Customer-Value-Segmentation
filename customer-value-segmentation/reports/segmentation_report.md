# Segmentation Report

*Report generated 2026-09-06 from a live run of the analysis pipeline (`src/run_pipeline.py`). All figures below are computed values, not illustrative placeholders.*

## 1. Data Preparation Summary

- Raw customers generated: **10,532** → after cleaning: **10,500**
- Raw transactions generated: **112,055** → after cleaning: **110,610**
- Duplicate customer rows removed: **32**
- Duplicate transaction rows removed: **889**
- Transaction rows dropped for missing `order_value` (could not be safely imputed): **556**
- Negative `order_value` entries corrected (data-entry sign errors): **8**
- Extreme order-value outliers identified via 3×IQR (winsorised, not dropped): **1526**

Full detail of every cleaning decision and its rationale is in `quality_control_report.md`.

## 2. RFM & Feature Engineering

Recency, Frequency and Monetary value were computed per customer relative to a snapshot date of 2025-01-31, alongside behavioural features (distinct categories purchased, return rate, tenure, engagement score, spend/orders normalised by tenure). See `data_dictionary.md` for full definitions.

- Customers with **zero purchases in the window**: 1,204 (11.47% of the base) — these are retained as a real, low-recency-by-construction group rather than dropped, since "never purchased" is itself a meaningful business state.

## 3. Exploratory Analysis

**Skewness of core variables (raw scale):**

| Variable | Skewness |
|---|---|
| `recency_days` | 2.69 |
| `frequency` | 1.02 |
| `monetary_total` | 1.45 |
| `avg_order_value` | 0.17 |
| `tenure_months` | 0.19 |

Recency, Frequency and Monetary are all right-skewed (Monetary and Recency particularly so), which is expected for retail transaction data — most customers purchase infrequently/modestly, with a long tail of high-activity customers. A Yeo-Johnson power transform was applied to skewed features before scaling (see below) to prevent the tail from dominating Euclidean-distance-based clustering.

**Feature redundancy** (Pearson |r| > 0.8):

| Feature A | Feature B | r |
|---|---|---|
| `frequency` | `monetary_total` | 0.97 |
| `frequency` | `distinct_categories_purchased` | 0.888 |
| `monetary_total` | `distinct_categories_purchased` | 0.803 |
| `avg_order_value` | `median_order_value` | 0.974 |
| `orders_per_tenure_month` | `spend_per_tenure_month` | 0.904 |

`frequency` and `monetary_total` are highly correlated, as expected (more orders → more total spend). Both were still retained in the clustering feature set because they carry distinct business meaning (purchase *volume* vs purchase *value* — a customer who buys often at low value is strategically different from one who buys rarely at high value), and dropping either would remove information the business explicitly cares about. `avg_order_value` and `median_order_value` are similarly redundant; only `avg_order_value` was retained in the final clustering feature set to avoid double-counting basket size.

See `visualisations/01_rfm_distributions.png` and `visualisations/02_correlation_matrix.png`.

## 4. Preprocessing for Clustering

Final feature set used for clustering (10 features): `recency_days`, `frequency`, `monetary_total`, `avg_order_value`, `distinct_categories_purchased`, `return_rate`, `tenure_months`, `orders_per_tenure_month`, `spend_per_tenure_month`, `engagement_score`.

Steps applied, in order:
1. **Yeo-Johnson power transform** on right-skewed features (handles zero/negative values, unlike log1p, and normalises both the shape and scale of the distribution).
2. **Standardisation** (zero mean, unit variance) on all features. This is essential for distance-based algorithms like K-Means and Ward-linkage hierarchical clustering: without it, `monetary_total` (scale: £0-thousands) would dominate the distance calculation over, say, `return_rate` (scale: 0-1), regardless of each feature's actual business importance.

## 5. Cluster Count Selection

K-Means and Agglomerative (Ward-linkage) hierarchical clustering were compared across k = 2–8 using the elbow method (inertia), mean silhouette score, Davies-Bouldin index, bootstrap stability (mean Adjusted Rand Index across 10 subsamples), and agreement between the two algorithms (ARI).

|   k |   inertia |   kmeans_silhouette |   hierarchical_silhouette |   kmeans_davies_bouldin |   kmeans_vs_hier_agreement_ari |   bootstrap_stability_ari |
|----:|----------:|--------------------:|--------------------------:|------------------------:|-------------------------------:|--------------------------:|
|   2 |   64393.8 |               0.325 |                     0.288 |                   1.188 |                          0.615 |                     0.987 |
|   3 |   53128.1 |               0.259 |                     0.248 |                   1.336 |                          0.65  |                     0.995 |
|   4 |   46927.7 |               0.247 |                     0.151 |                   1.523 |                          0.422 |                     0.978 |
|   5 |   41797.3 |               0.253 |                     0.161 |                   1.384 |                          0.42  |                     0.977 |

**k = 4 was selected**, not purely because it produced the highest silhouette score. In fact, k=2 has the highest raw silhouette (0.3251) — but inspecting that partition shows it separates customers almost entirely into "has purchased" vs "has not purchased," which is too coarse to be strategically actionable. k=3 offers a slightly better silhouette/stability trade-off than k=4, but collapses two behaviourally and strategically distinct groups — newly acquired low-activity customers and long-tenured customers who have gone quiet — into a single mid-tier cluster, which would recommend the same treatment (e.g. onboarding vs win-back) to customers who need opposite interventions. k=4 is the smallest cluster count that cleanly separates these groups while keeping bootstrap stability high (mean ARI = 0.9778) and silhouette respectable (0.2471). Beyond k=4, additional clusters (k=5+) mainly subdivide existing groups without materially improving separation (see the elbow curve flattening from k=4 onward) or producing new business-actionable distinctions.

K-Means was chosen as the primary algorithm over hierarchical clustering: at the selected k, K-Means achieves a higher silhouette score (0.2471 vs 0.1507) and lower Davies-Bouldin index, and scales better for a 10,000+ row production dataset. The two algorithms agree only moderately at this k (ARI = 0.4216) — hierarchical clustering with Ward linkage tends to produce more evenly-sized clusters and is more influenced by chain-like structure in the transformed feature space, whereas K-Means finds more centroid-compact groups; this is a known, expected behavioural difference between the two methods rather than evidence that either is wrong. The hierarchical dendrogram (`visualisations/00_dendrogram_sample.png`) is retained as supporting evidence.

See `visualisations/03_elbow_curve.png` and `visualisations/04_silhouette_by_k.png`.

## 6. Dimensionality Reduction (PCA)

PCA was applied purely for 2D visualisation of the 4-cluster solution (`visualisations/05_pca_clusters.png`). The first two components explain 56.2% and 13.3% of variance respectively (69.5% combined). Note that PCA components are linear combinations of all standardised input features and **do not correspond to a single business metric** — PC1 and PC2 should be read as "directions of maximum variance in the customer-behaviour feature space," not as, say, "value" and "engagement" axes. The clustering itself was performed on the full standardised feature space, not on the PCA-reduced data — PCA is a visualisation aid only.

## 7. Final Segments

### High-Value Loyal — 3,816 customers (36.34% of base, 83.3% of total spend)

- Average recency: **29.9 days** since last order
- Average frequency: **21.75 orders**
- Average monetary value: **£1,347.58**
- Average order value: **£59.41**
- Average tenure: **18.5 months**
- Average distinct categories purchased: **8.71**
- Average return rate: **7.0%**

### Emerging Customers — 2,228 customers (21.22% of base, 8.4% of total spend)

- Average recency: **27.0 days** since last order
- Average frequency: **5.89 orders**
- Average monetary value: **£232.86**
- Average order value: **£40.11**
- Average tenure: **5.7 months**
- Average distinct categories purchased: **4.20**
- Average return rate: **6.0%**

### At-Risk / Lapsing — 2,992 customers (28.5% of base, 8.23% of total spend)

- Average recency: **127.4 days** since last order
- Average frequency: **4.75 orders**
- Average monetary value: **£169.77**
- Average order value: **£35.11**
- Average tenure: **13.9 months**
- Average distinct categories purchased: **3.65**
- Average return rate: **11.0%**

### New / Low Engagement — 1,464 customers (13.94% of base, 0.07% of total spend)

- Average recency: **105.1 days** since last order
- Average frequency: **0.21 orders**
- Average monetary value: **£2.95**
- Average order value: **£2.53**
- Average tenure: **3.9 months**
- Average distinct categories purchased: **0.20**
- Average return rate: **0.0%**

See `visualisations/06_segment_size.png`, `07_segment_value.png`, `08_behavioural_profiles.png`, and `09_segment_comparison.png` for the supporting charts, and `visualisations/10_executive_dashboard.png` for the summary view.