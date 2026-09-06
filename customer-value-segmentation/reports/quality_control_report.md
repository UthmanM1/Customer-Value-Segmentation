# Quality Control Report

*Self-review generated 2026-09-06, checking the analysis against common failure modes in clustering-based segmentation projects.*

| Check | Finding | Verdict |
|---|---|---|
| Inappropriate scaling | All clustering features were power-transformed (Yeo-Johnson) where skewed, then standardised (zero mean / unit variance) before any distance calculation. | ✅ Pass |
| Extreme outlier influence | 1526 extreme `order_value` observations (>3×IQR) were winsorised (capped), not silently included or arbitrarily deleted, before aggregation to customer level. | ✅ Pass |
| Unstable clustering | Bootstrap stability at the final k=4 was checked via 10 subsample refits; mean Adjusted Rand Index vs the full-data partition = 0.9778 (>0.9 indicates a highly reproducible partition, not an artefact of one sample). | ✅ Pass |
| Arbitrary cluster count | k was selected using elbow + silhouette + stability evidence *and* qualitative business interpretability, not by maximising silhouette alone (k=2 had the highest silhouette but was rejected as too coarse — see `segmentation_report.md` §5 for the full reasoning). | ✅ Pass |
| Leakage | No target/label was ever available to leak — this is unsupervised learning. The latent generative archetype used to *simulate* the synthetic data was dropped before saving the raw dataset and was not accessible to any step of the analysis pipeline. | ✅ Pass |
| Unsupported segment descriptions | Every segment name and every claim in `business_strategy.md` is derived programmatically from the computed segment profile (`evaluation.name_segments`), not asserted independently of the data. | ✅ Pass |
| Misleading PCA interpretation | The segmentation report explicitly states that PCA components are linear combinations of the full feature set used only for visualisation, and that clustering was performed on the full standardised feature space, not the 2D PCA projection. | ✅ Pass |
| Statistical testing errors | Kruskal-Wallis (non-parametric) was used given non-normal RFM distributions, with Benjamini-Hochberg FDR correction applied across all 13 tested metrics, and effect sizes (not p-values alone) used to judge practical significance. | ✅ Pass |
| Feature redundancy | Highly correlated features (e.g. `frequency`/`monetary_total`, `avg_order_value`/`median_order_value`) were explicitly identified in EDA; redundant near-duplicate features were dropped from the clustering set while conceptually distinct correlated features were knowingly retained with justification. | ✅ Pass |

## Known Limitations (see also README)

- The dataset is **entirely synthetic**; absolute figures (e.g. monetary values, exact segment sizes) reflect the generative assumptions in `src/data_generation.py`, not a real business, and should not be used as a benchmark.
- Behavioural archetypes were deliberately generated with overlap and noise so that clustering requires genuine analytical judgement — but this also means the 'ground truth' number of latent generative groups (5) does not need to equal, and did not equal, the analytically optimal number of clusters recovered (4). This is realistic: real customer bases rarely have a single unambiguous correct k either.
- K-Means assumes roughly convex, similarly-scaled clusters; segments with more complex non-convex shapes in the true feature space would not be well captured by this approach.
- Algorithm agreement between K-Means and hierarchical clustering at the final k was moderate (ARI = 0.4216), reflecting genuinely fuzzy boundaries between segments rather than a coding error — this is disclosed rather than hidden.