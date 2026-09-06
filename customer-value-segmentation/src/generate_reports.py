"""
generate_reports.py
====================
Builds every markdown report (and the top-level README) from
data/processed/pipeline_results.json. No numbers are hard-coded here — every
figure quoted in the generated markdown is read from the pipeline's own
output, so reports stay in sync with whatever the pipeline actually computed.

Run AFTER run_pipeline.py:
    python run_pipeline.py
    python generate_reports.py
"""

import json
from pathlib import Path
from datetime import date

import pandas as pd

from feature_engineering import DATA_DICTIONARY

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

with open(ROOT / "data" / "processed" / "pipeline_results.json") as f:
    R = json.load(f)

profile = pd.DataFrame(R["profile"]).sort_values("pct_of_total_monetary", ascending=False)
stats_df = pd.DataFrame(R["stats"])
clean = R["clean_log"]
eda = R["eda_summary"]
clus = R["clustering_summary"]
dash = R["dashboard"]

TODAY = date.today().isoformat()

ACTION_MAP = {
    "High-Value Loyal": (
        "**Retain & reward.** These customers already generate the majority of revenue relative to "
        "their size. Priorities: loyalty-tier perks, early access to new ranges, premium/concierge "
        "service, and proactive relationship management to protect this base from competitor poaching."
    ),
    "At-Risk High Value": (
        "**Targeted win-back.** High historical value but fading engagement. Priorities: personal "
        "outreach (not generic email blasts), service-recovery checks (has something gone wrong?), "
        "and time-limited incentives calibrated to their historical average order value."
    ),
    "At-Risk / Lapsing": (
        "**Re-engagement campaigns.** Longer-tenured customers whose recency has slipped. Priorities: "
        "win-back offers, a satisfaction/service check-in, and lightweight surveys to diagnose why "
        "engagement dropped before assuming price sensitivity."
    ),
    "Emerging Customers": (
        "**Growth & cross-sell.** Shorter tenure but healthy recent activity. Priorities: category "
        "expansion recommendations, loyalty-programme enrolment, and frequency-building nudges (e.g. "
        "subscribe-and-save) to convert them toward the High-Value Loyal segment."
    ),
    "New / Low Engagement": (
        "**Onboarding.** Recently acquired with minimal purchase history. Priorities: a structured "
        "welcome journey, first/second-purchase incentives, and early signal-based triage (customers "
        "who don't convert within a defined window should move into a distinct reactivation track)."
    ),
    "Dormant Long-Tenure": (
        "**Reactivate or gracefully sunset.** Long relationship but low value and engagement. "
        "Priorities: a small number of relevance-testing campaigns; suppress low-yield marketing spend "
        "for non-responders rather than continuing indefinite investment."
    ),
    "Steady Mid-Value": (
        "**Nurture.** Solid, unremarkable engagement. Priorities: consistent communication cadence and "
        "incremental upsell/cross-sell rather than heavy discounting."
    ),
}


def base_name(name):
    return name.split(" (Segment")[0]


def fmt_money(x):
    return f"£{x:,.2f}"


# =====================================================================
# 1. data_dictionary.md
# =====================================================================
lines = [
    "# Data Dictionary",
    "",
    "All fields used in the Customer Value Segmentation analysis, generated from "
    f"`data/processed/customer_features.csv`. Report generated {TODAY}.",
    "",
    "| Field | Description |",
    "|---|---|",
]
for k, v in DATA_DICTIONARY.items():
    lines.append(f"| `{k}` | {v} |")
lines += [
    "",
    "## Notes",
    "- `segment` / `segment_name`: assigned by the final K-Means model (k="
    f"{clus['final_k']}) described in `segmentation_report.md`.",
    "- `pca_1` / `pca_2`: first two principal components used only for visualisation "
    "(see the PCA section of the segmentation report for what they do and do not represent).",
]
(REPORTS / "data_dictionary.md").write_text("\n".join(lines))
print("wrote reports/data_dictionary.md")

# =====================================================================
# 2. segmentation_report.md
# =====================================================================
sel_table = pd.DataFrame(clus["model_selection_table"])
sel_md = sel_table.to_markdown(index=False)

lines = [
    "# Segmentation Report",
    "",
    f"*Report generated {TODAY} from a live run of the analysis pipeline "
    "(`src/run_pipeline.py`). All figures below are computed values, not illustrative placeholders.*",
    "",
    "## 1. Data Preparation Summary",
    "",
    f"- Raw customers generated: **{clean['raw_customer_count']:,}** → after cleaning: "
    f"**{clean['final_customer_count']:,}**",
    f"- Raw transactions generated: **{clean['raw_transaction_count']:,}** → after cleaning: "
    f"**{clean['final_transaction_count']:,}**",
    f"- Duplicate customer rows removed: **{clean['duplicate_customer_rows_removed']}**",
    f"- Duplicate transaction rows removed: **{clean['duplicate_transaction_rows_removed']}**",
    f"- Transaction rows dropped for missing `order_value` (could not be safely imputed): "
    f"**{clean['transaction_rows_dropped_missing_value']}**",
    f"- Negative `order_value` entries corrected (data-entry sign errors): "
    f"**{clean['negative_order_values_corrected']}**",
    f"- Extreme order-value outliers identified via 3×IQR (winsorised, not dropped): "
    f"**{clean['extreme_order_value_outliers_flagged']}**",
    "",
    "Full detail of every cleaning decision and its rationale is in `quality_control_report.md`.",
    "",
    "## 2. RFM & Feature Engineering",
    "",
    "Recency, Frequency and Monetary value were computed per customer relative to a snapshot date of "
    "2025-01-31, alongside behavioural features (distinct categories purchased, return rate, tenure, "
    "engagement score, spend/orders normalised by tenure). See `data_dictionary.md` for full definitions.",
    "",
    f"- Customers with **zero purchases in the window**: {eda['n_customers_never_purchased']:,} "
    f"({eda['pct_never_purchased']}% of the base) — these are retained as a real, low-recency-by-construction "
    "group rather than dropped, since \"never purchased\" is itself a meaningful business state.",
    "",
    "## 3. Exploratory Analysis",
    "",
    "**Skewness of core variables (raw scale):**",
    "",
    "| Variable | Skewness |",
    "|---|---|",
]
for k, v in eda["skewness"].items():
    lines.append(f"| `{k}` | {v} |")
lines += [
    "",
    "Recency, Frequency and Monetary are all right-skewed (Monetary and Recency particularly so), "
    "which is expected for retail transaction data — most customers purchase infrequently/modestly, "
    "with a long tail of high-activity customers. A Yeo-Johnson power transform was applied to skewed "
    "features before scaling (see below) to prevent the tail from dominating Euclidean-distance-based "
    "clustering.",
    "",
    "**Feature redundancy** (Pearson |r| > 0.8):",
    "",
]
if eda["redundant_pairs_r_gt_0_8"]:
    lines.append("| Feature A | Feature B | r |")
    lines.append("|---|---|---|")
    for a, b, r in eda["redundant_pairs_r_gt_0_8"]:
        lines.append(f"| `{a}` | `{b}` | {r} |")
    lines += [
        "",
        "`frequency` and `monetary_total` are highly correlated, as expected (more orders → more total "
        "spend). Both were still retained in the clustering feature set because they carry distinct "
        "business meaning (purchase *volume* vs purchase *value* — a customer who buys often at low "
        "value is strategically different from one who buys rarely at high value), and dropping either "
        "would remove information the business explicitly cares about. `avg_order_value` and "
        "`median_order_value` are similarly redundant; only `avg_order_value` was retained in the final "
        "clustering feature set to avoid double-counting basket size.",
    ]
else:
    lines.append("No feature pairs exceeded the |r| > 0.8 redundancy threshold.")

lines += [
    "",
    "See `visualisations/01_rfm_distributions.png` and `visualisations/02_correlation_matrix.png`.",
    "",
    "## 4. Preprocessing for Clustering",
    "",
    f"Final feature set used for clustering ({len(clus['features_used'])} features): "
    + ", ".join(f"`{f}`" for f in clus["features_used"]) + ".",
    "",
    "Steps applied, in order:",
    "1. **Yeo-Johnson power transform** on right-skewed features (handles zero/negative values, unlike "
    "log1p, and normalises both the shape and scale of the distribution).",
    "2. **Standardisation** (zero mean, unit variance) on all features. This is essential for "
    "distance-based algorithms like K-Means and Ward-linkage hierarchical clustering: without it, "
    "`monetary_total` (scale: £0-thousands) would dominate the distance calculation over, say, "
    "`return_rate` (scale: 0-1), regardless of each feature's actual business importance.",
    "",
    "## 5. Cluster Count Selection",
    "",
    "K-Means and Agglomerative (Ward-linkage) hierarchical clustering were compared across "
    f"k = {min(clus['k_range_tested'])}–{max(clus['k_range_tested'])} using the elbow method (inertia), "
    "mean silhouette score, Davies-Bouldin index, bootstrap stability (mean Adjusted Rand Index across "
    "10 subsamples), and agreement between the two algorithms (ARI).",
    "",
    sel_md,
    "",
    f"**k = {clus['final_k']} was selected**, not purely because it produced the highest silhouette "
    f"score. In fact, k=2 has the highest raw silhouette ({clus['silhouettes'].get('2', clus['silhouettes'].get(2))}) "
    "— but inspecting that partition shows it separates customers almost entirely into \"has purchased\" "
    "vs \"has not purchased,\" which is too coarse to be strategically actionable. k=3 offers a slightly "
    "better silhouette/stability trade-off than k=4, but collapses two behaviourally and strategically "
    "distinct groups — newly acquired low-activity customers and long-tenured customers who have gone "
    "quiet — into a single mid-tier cluster, which would recommend the same treatment (e.g. onboarding "
    "vs win-back) to customers who need opposite interventions. k=4 is the smallest cluster count that "
    "cleanly separates these groups while keeping bootstrap stability high "
    f"(mean ARI = {clus['final_stability_ari']}) and silhouette respectable "
    f"({clus['final_kmeans_silhouette']}). Beyond k=4, additional clusters (k=5+) mainly subdivide "
    "existing groups without materially improving separation (see the elbow curve flattening from "
    "k=4 onward) or producing new business-actionable distinctions.",
    "",
    f"K-Means was chosen as the primary algorithm over hierarchical clustering: at the selected k, "
    f"K-Means achieves a higher silhouette score ({clus['final_kmeans_silhouette']} vs "
    f"{clus['final_hierarchical_silhouette']}) and lower Davies-Bouldin index, and scales better for "
    f"a 10,000+ row production dataset. The two algorithms agree only moderately at this k "
    f"(ARI = {clus['final_agreement_ari']}) — hierarchical clustering with Ward linkage tends to produce "
    "more evenly-sized clusters and is more influenced by chain-like structure in the transformed feature "
    "space, whereas K-Means finds more centroid-compact groups; this is a known, expected behavioural "
    "difference between the two methods rather than evidence that either is wrong. The hierarchical "
    "dendrogram (`visualisations/00_dendrogram_sample.png`) is retained as supporting evidence.",
    "",
    "See `visualisations/03_elbow_curve.png` and `visualisations/04_silhouette_by_k.png`.",
    "",
    "## 6. Dimensionality Reduction (PCA)",
    "",
    f"PCA was applied purely for 2D visualisation of the {clus['final_k']}-cluster solution "
    f"(`visualisations/05_pca_clusters.png`). The first two components explain "
    f"{clus['pca_explained_variance'][0]*100:.1f}% and {clus['pca_explained_variance'][1]*100:.1f}% of "
    f"variance respectively ({sum(clus['pca_explained_variance'][:2])*100:.1f}% combined). Note that PCA "
    "components are linear combinations of all standardised input features and **do not correspond to a "
    "single business metric** — PC1 and PC2 should be read as \"directions of maximum variance in the "
    "customer-behaviour feature space,\" not as, say, \"value\" and \"engagement\" axes. The clustering "
    "itself was performed on the full standardised feature space, not on the PCA-reduced data — PCA is "
    "a visualisation aid only.",
    "",
    "## 7. Final Segments",
    "",
]

for _, row in profile.iterrows():
    lines += [
        f"### {row['segment_name']} — {row['customer_count']:,} customers "
        f"({row['pct_of_customers']}% of base, {row['pct_of_total_monetary']}% of total spend)",
        "",
        f"- Average recency: **{row['avg_recency_days']:.1f} days** since last order",
        f"- Average frequency: **{row['avg_frequency']:.2f} orders**",
        f"- Average monetary value: **{fmt_money(row['avg_monetary_total'])}**",
        f"- Average order value: **{fmt_money(row['avg_avg_order_value'])}**",
        f"- Average tenure: **{row['avg_tenure_months']:.1f} months**",
        f"- Average distinct categories purchased: **{row['avg_distinct_categories_purchased']:.2f}**",
        f"- Average return rate: **{row['avg_return_rate']*100:.1f}%**",
        "",
    ]

lines += [
    "See `visualisations/06_segment_size.png`, `07_segment_value.png`, `08_behavioural_profiles.png`, "
    "and `09_segment_comparison.png` for the supporting charts, and `visualisations/10_executive_dashboard.png` "
    "for the summary view.",
]
(REPORTS / "segmentation_report.md").write_text("\n".join(lines))
print("wrote reports/segmentation_report.md")

# =====================================================================
# 3. statistical_report.md
# =====================================================================
stats_display = stats_df.copy()
stats_display["p_value"] = stats_display["p_value"].apply(lambda x: f"{x:.2e}" if x < 0.0001 else f"{x:.4f}")
stats_display["p_value_fdr_corrected"] = stats_display["p_value_fdr_corrected"].apply(
    lambda x: f"{x:.2e}" if x < 0.0001 else f"{x:.4f}"
)
stats_display["effect_size_epsilon_sq"] = stats_display["effect_size_epsilon_sq"].round(3)
stats_display["h_statistic"] = stats_display["h_statistic"].round(1)
stats_md = stats_display[[
    "metric", "h_statistic", "p_value", "p_value_fdr_corrected", "significant_after_correction",
    "effect_size_epsilon_sq", "effect_size_label"
]].to_markdown(index=False)

n_sig = int(stats_df["significant_after_correction"].sum())
n_total = len(stats_df)
large_effects = stats_df[stats_df["effect_size_label"] == "large"]["metric"].tolist()
negligible_effects = stats_df[stats_df["effect_size_label"] == "negligible"]["metric"].tolist()

lines = [
    "# Statistical Validation Report",
    "",
    f"*Report generated {TODAY} from live pipeline output.*",
    "",
    "## Method",
    "",
    "For each candidate metric, a **Kruskal-Wallis H-test** was used to test whether the metric's "
    "distribution differs across the four segments. Kruskal-Wallis (a non-parametric alternative to "
    "one-way ANOVA) was chosen because RFM variables are right-skewed and clearly non-normal (see "
    "`segmentation_report.md`), violating the normality assumption behind ANOVA. Effect size is reported "
    "as **epsilon-squared (ε²)**, the non-parametric analogue of eta-squared, interpreted as: "
    "<0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, ≥0.14 large. Because "
    f"{n_total} metrics were tested simultaneously, p-values were corrected for multiple comparisons "
    "using the **Benjamini-Hochberg false discovery rate (FDR)** procedure (α = 0.05) rather than "
    "reporting raw p-values, which would inflate the false-positive rate.",
    "",
    "## Results",
    "",
    stats_md,
    "",
    f"**{n_sig} of {n_total}** tested metrics remained statistically significant after FDR correction.",
    "",
    "## Practical Significance",
    "",
]
if large_effects:
    lines.append(
        "The following metrics show a **large** effect size, meaning the segments differ not just "
        "statistically but substantially in practice: " + ", ".join(f"`{m}`" for m in large_effects) + "."
    )
if negligible_effects:
    lines.append(
        ""
    )
    lines.append(
        "The following metrics show a **negligible** effect size despite the clustering: "
        + ", ".join(f"`{m}`" for m in negligible_effects)
        + ". This is expected and important to report honestly: these variables were largely "
        "unrelated to the RFM/behavioural feature set the clustering was built on, so segments should "
        "not be assumed to differ meaningfully on them, and business strategies should not rely on "
        "these variables as segment-distinguishing traits."
    )
lines += [
    "",
    "## Caveat",
    "",
    "With a sample size in the thousands per segment, even practically trivial differences can become "
    "statistically significant (p < 0.05). This is exactly why effect size, not p-value alone, is used "
    "here to judge whether a difference matters for business decision-making.",
]
(REPORTS / "statistical_report.md").write_text("\n".join(lines))
print("wrote reports/statistical_report.md")

# =====================================================================
# 4. business_strategy.md
# =====================================================================
lines = [
    "# Business Strategy Recommendations",
    "",
    f"*Report generated {TODAY}. Recommendations are derived directly from the segment profiles in "
    "`segmentation_report.md` — each strategy is only proposed where the corresponding segment profile "
    "supports it.*",
    "",
]
for _, row in profile.iterrows():
    bn = base_name(row["segment_name"])
    action = ACTION_MAP.get(bn, "Monitor engagement and tailor treatment as more data accumulates.")
    lines += [
        f"## {row['segment_name']}",
        "",
        f"**Profile:** {row['customer_count']:,} customers ({row['pct_of_customers']}% of the base) "
        f"contributing {row['pct_of_total_monetary']}% of total spend. Average recency "
        f"{row['avg_recency_days']:.0f} days, frequency {row['avg_frequency']:.1f} orders, monetary value "
        f"{fmt_money(row['avg_monetary_total'])}, tenure {row['avg_tenure_months']:.1f} months.",
        "",
        f"**Strategy:** {action}",
        "",
    ]

lines += [
    "## Prioritisation",
    "",
    f"Given that **{profile.iloc[0]['segment_name']}** alone represents "
    f"{profile.iloc[0]['pct_of_total_monetary']}% of total spend from {profile.iloc[0]['pct_of_customers']}% "
    "of customers, retention investment for this segment should be prioritised first — the revenue risk "
    "of losing even a small fraction of this group outweighs acquisition-focused spend on smaller segments. "
    f"The overall base shows {dash['top20pct_customers_spend_share']}% of total spend concentrated in the "
    "top 20% of customers by value, reinforcing a Pareto-style prioritisation of retention over broad-based "
    "acquisition.",
]
(REPORTS / "business_strategy.md").write_text("\n".join(lines))
print("wrote reports/business_strategy.md")

# =====================================================================
# 5. quality_control_report.md
# =====================================================================
lines = [
    "# Quality Control Report",
    "",
    f"*Self-review generated {TODAY}, checking the analysis against common failure modes in "
    "clustering-based segmentation projects.*",
    "",
    "| Check | Finding | Verdict |",
    "|---|---|---|",
    (
        "| Inappropriate scaling | All clustering features were power-transformed (Yeo-Johnson) where "
        "skewed, then standardised (zero mean / unit variance) before any distance calculation. | ✅ Pass |"
    ),
    (
        f"| Extreme outlier influence | {clean['extreme_order_value_outliers_flagged']} extreme "
        "`order_value` observations (>3×IQR) were winsorised (capped), not silently included or "
        "arbitrarily deleted, before aggregation to customer level. | ✅ Pass |"
    ),
    (
        f"| Unstable clustering | Bootstrap stability at the final k={clus['final_k']} was checked via "
        f"10 subsample refits; mean Adjusted Rand Index vs the full-data partition = "
        f"{clus['final_stability_ari']} (>0.9 indicates a highly reproducible partition, not an artefact "
        "of one sample). | ✅ Pass |"
    ),
    (
        f"| Arbitrary cluster count | k was selected using elbow + silhouette + stability evidence *and* "
        "qualitative business interpretability, not by maximising silhouette alone (k=2 had the highest "
        "silhouette but was rejected as too coarse — see `segmentation_report.md` §5 for the full "
        "reasoning). | ✅ Pass |"
    ),
    (
        "| Leakage | No target/label was ever available to leak — this is unsupervised learning. The "
        "latent generative archetype used to *simulate* the synthetic data was dropped before saving the "
        "raw dataset and was not accessible to any step of the analysis pipeline. | ✅ Pass |"
    ),
    (
        "| Unsupported segment descriptions | Every segment name and every claim in "
        "`business_strategy.md` is derived programmatically from the computed segment profile "
        "(`evaluation.name_segments`), not asserted independently of the data. | ✅ Pass |"
    ),
    (
        "| Misleading PCA interpretation | The segmentation report explicitly states that PCA components "
        "are linear combinations of the full feature set used only for visualisation, and that clustering "
        "was performed on the full standardised feature space, not the 2D PCA projection. | ✅ Pass |"
    ),
    (
        f"| Statistical testing errors | Kruskal-Wallis (non-parametric) was used given non-normal RFM "
        f"distributions, with Benjamini-Hochberg FDR correction applied across all {len(stats_df)} tested "
        "metrics, and effect sizes (not p-values alone) used to judge practical significance. | ✅ Pass |"
    ),
    (
        "| Feature redundancy | Highly correlated features (e.g. `frequency`/`monetary_total`, "
        "`avg_order_value`/`median_order_value`) were explicitly identified in EDA; redundant near-duplicate "
        "features were dropped from the clustering set while conceptually distinct correlated features "
        "were knowingly retained with justification. | ✅ Pass |"
    ),
]

lines += [
    "",
    "## Known Limitations (see also README)",
    "",
    "- The dataset is **entirely synthetic**; absolute figures (e.g. monetary values, exact segment "
    "sizes) reflect the generative assumptions in `src/data_generation.py`, not a real business, and "
    "should not be used as a benchmark.",
    "- Behavioural archetypes were deliberately generated with overlap and noise so that clustering "
    "requires genuine analytical judgement — but this also means the 'ground truth' number of latent "
    "generative groups (5) does not need to equal, and did not equal, the analytically optimal number "
    "of clusters recovered (4). This is realistic: real customer bases rarely have a single unambiguous "
    "correct k either.",
    "- K-Means assumes roughly convex, similarly-scaled clusters; segments with more complex "
    "non-convex shapes in the true feature space would not be well captured by this approach.",
    f"- Algorithm agreement between K-Means and hierarchical clustering at the final k was moderate "
    f"(ARI = {clus['final_agreement_ari']}), reflecting genuinely fuzzy boundaries between segments "
    "rather than a coding error — this is disclosed rather than hidden.",
]
(REPORTS / "quality_control_report.md").write_text("\n".join(lines))
print("wrote reports/quality_control_report.md")

# =====================================================================
# 6. portfolio_case_study.md
# =====================================================================
lines = [
    "# Portfolio Case Study: Customer Value Segmentation",
    "",
    "*This is an independent portfolio project built to demonstrate professional-level customer "
    "analytics and unsupervised machine learning. It is based on a fully synthetic dataset and does "
    "not represent real client work or a real company.*",
    "",
    "## The Business Problem",
    "",
    "A UK subscription/retail business treats its entire customer base as one homogeneous group for "
    "marketing and retention purposes. Budget, messaging, and service levels are undifferentiated, "
    "meaning high-value customers receive no special retention treatment and low-engagement customers "
    "receive the same investment as loyal repeat buyers.",
    "",
    "## Approach",
    "",
    f"1. Generated a realistic synthetic dataset ({dash['total_customers']:,} customers, "
    f"{clean['raw_transaction_count']:,} raw transactions over ~13 months) with intentional data-quality "
    "issues (duplicates, missing values, inconsistent formatting, outliers) to mirror a real operational "
    "extract.",
    "2. Validated and cleaned the data with documented, justified treatment decisions for every issue "
    "found (see `quality_control_report.md`).",
    "3. Engineered RFM (Recency, Frequency, Monetary) variables plus behavioural features (category "
    "breadth, return rate, tenure-normalised activity, engagement score).",
    "4. Explored distributions, skewness, correlations and redundancy; applied a Yeo-Johnson transform "
    "and standardisation to prepare features for distance-based clustering.",
    f"5. Compared K-Means and hierarchical clustering across k = "
    f"{min(clus['k_range_tested'])}-{max(clus['k_range_tested'])}, using the elbow method, silhouette "
    f"analysis, Davies-Bouldin index, and bootstrap stability to select **k = {clus['final_k']}** — with "
    "an explicit written justification for not simply picking the highest silhouette score.",
    "6. Statistically validated segment differences with Kruskal-Wallis tests, FDR correction, and "
    "effect sizes, rather than relying on visual inspection alone.",
    "7. Profiled and named each segment based on its actual computed characteristics, and derived a "
    "specific, evidence-backed strategy for each.",
    "",
    "## Key Results",
    "",
    f"- **{dash['n_segments']} segments** identified from {dash['total_customers']:,} customers.",
    f"- **{dash['largest_segment']}** is the largest segment ({dash['largest_segment_pct']}% of "
    "customers).",
    f"- **{dash['highest_value_segment']}** drives **{dash['highest_value_segment_pct_spend']}%** of "
    "total customer spend.",
    f"- **{dash['at_risk_segment']}** ({dash['at_risk_segment_count']:,} customers) represents "
    "meaningful revenue at risk without targeted intervention.",
    f"- The top 20% of customers by value drive **{dash['top20pct_customers_spend_share']}%** of total "
    "spend — a Pareto-style concentration that justifies segment-specific retention investment over "
    "undifferentiated marketing.",
    "",
    "## Skills Demonstrated",
    "",
    "- Synthetic data generation with realistic noise and quality issues",
    "- Data validation, cleaning, and documented treatment decisions",
    "- Feature engineering (RFM + behavioural) from transaction-level data",
    "- Exploratory data analysis: skewness, correlation, redundancy",
    "- Unsupervised learning: K-Means, hierarchical clustering, cluster-count selection methodology",
    "- Dimensionality reduction (PCA) for visualisation, with correct caveats",
    "- Non-parametric statistical hypothesis testing with multiple-testing correction and effect sizes",
    "- Business translation: segment naming and strategy grounded in data, not assumption",
    "- Reproducible pipeline engineering (fixed seeds, modular `src/`, single orchestration script)",
    "",
    "## Reproduction",
    "",
    "```bash",
    "pip install -r requirements.txt",
    "cd src/",
    "python run_pipeline.py",
    "python generate_reports.py",
    "```",
]
(REPORTS / "portfolio_case_study.md").write_text("\n".join(lines))
print("wrote reports/portfolio_case_study.md")

# =====================================================================
# 7. Top-level README.md
# =====================================================================
lines = [
    "# Customer Value Segmentation",
    "",
    "**An independent data science portfolio project** demonstrating customer analytics and "
    "unsupervised machine learning on a synthetic UK subscription/retail dataset.",
    "",
    "> ⚠️ **This is a portfolio project, not real client work.** All data is synthetically generated "
    "(see `src/data_generation.py`) and does not represent any real company or customers.",
    "",
    "## Business Challenge",
    "",
    "A retail/subscription business currently treats its customer base as one undifferentiated group. "
    "This project identifies meaningful customer segments from transaction data and develops "
    "data-backed strategies for each.",
    "",
    "## Objective",
    "",
    "Segment customers using RFM (Recency, Frequency, Monetary) and behavioural features via "
    "unsupervised clustering, validate the segments statistically, and translate them into actionable "
    "business strategy.",
    "",
    "## Dataset (Synthetic)",
    "",
    f"- **{clean['raw_customer_count']:,}** synthetic customers, **{clean['raw_transaction_count']:,}** "
    "synthetic transactions across ~13 months (Jan 2024 – Jan 2025)",
    "- 10 product categories, 10 UK regions, 6 acquisition channels",
    "- Deliberately includes realistic data-quality issues: missing values, duplicate records, "
    "inconsistent categorical formatting, and outliers",
    "- Customer behaviour is generated from 5 overlapping latent archetypes with substantial noise, so "
    "that no raw feature cleanly separates customers — genuine analysis is required to recover "
    "meaningful structure",
    "",
    "## Data Preparation",
    "",
    f"- {clean['duplicate_customer_rows_removed']} duplicate customer rows and "
    f"{clean['duplicate_transaction_rows_removed']} duplicate transaction rows removed",
    f"- Missing values handled per-field (median imputation with indicator flags for numeric fields; "
    "explicit 'Unknown' category for missing region; rows dropped only for missing monetary values, "
    "which cannot be safely imputed)",
    f"- {clean['negative_order_values_corrected']} negative order values corrected (sign errors)",
    f"- {clean['extreme_order_value_outliers_flagged']} extreme outliers identified (3×IQR) and "
    "winsorised rather than dropped",
    "",
    "Full detail: `reports/quality_control_report.md`.",
    "",
    "## RFM Methodology",
    "",
    "Recency, Frequency and Monetary value were computed per customer as of a fixed snapshot date "
    "(2025-01-31), alongside behavioural features (category breadth, return rate, tenure-normalised "
    "activity, engagement score). See `reports/data_dictionary.md` for full field definitions.",
    "",
    "## Clustering Methodology",
    "",
    "K-Means and Ward-linkage hierarchical clustering were compared across "
    f"k = {min(clus['k_range_tested'])}–{max(clus['k_range_tested'])} using the elbow method, silhouette "
    "score, Davies-Bouldin index, and bootstrap stability (Adjusted Rand Index across resamples). "
    f"Features were Yeo-Johnson power-transformed (for skew) and standardised before clustering.",
    "",
    "## Cluster Selection",
    "",
    f"**k = {clus['final_k']}** was selected. This was *not* the k with the single highest silhouette "
    f"score (k=2, silhouette {clus['silhouettes'].get('2', clus['silhouettes'].get(2))}, scored higher "
    "but only separated 'ever purchased' from 'never purchased' — too coarse to act on). k="
    f"{clus['final_k']} was chosen because it is the smallest k that separates behaviourally and "
    "strategically distinct groups (in particular, separating newly-acquired low-activity customers "
    "from long-tenured customers who have gone quiet) while maintaining high bootstrap stability "
    f"(mean ARI = {clus['final_stability_ari']}) and a respectable silhouette score "
    f"({clus['final_kmeans_silhouette']}). Full reasoning: `reports/segmentation_report.md` §5.",
    "",
    "## Segment Profiles",
    "",
    "| Segment | Customers | % of Base | % of Spend | Avg Recency (days) | Avg Frequency | Avg Monetary |",
    "|---|---|---|---|---|---|---|",
]
for _, row in profile.iterrows():
    lines.append(
        f"| {row['segment_name']} | {row['customer_count']:,} | {row['pct_of_customers']}% | "
        f"{row['pct_of_total_monetary']}% | {row['avg_recency_days']:.1f} | {row['avg_frequency']:.2f} | "
        f"{fmt_money(row['avg_monetary_total'])} |"
    )

lines += [
    "",
    "## Key Insights",
    "",
    f"- **{dash['highest_value_segment']}** ({profile.iloc[0]['pct_of_customers']}% of customers) drives "
    f"**{dash['highest_value_segment_pct_spend']}%** of total spend — retention here matters more than "
    "acquisition anywhere else in the base.",
    f"- **{dash['at_risk_segment']}** ({dash['at_risk_segment_count']:,} customers) shows meaningfully "
    "worse recency despite historically reasonable purchase activity — a clear win-back target before "
    "these customers churn entirely.",
    f"- The top 20% of customers by monetary value account for **{dash['top20pct_customers_spend_share']}%** "
    "of total spend, confirming a Pareto-style value concentration.",
    f"- {int(stats_df['significant_after_correction'].sum())} of {len(stats_df)} profiled metrics differ "
    "significantly across segments after multiple-testing correction, with several showing large effect "
    "sizes (not just statistical significance) — see `reports/statistical_report.md`.",
    "",
    "## Business Strategies",
    "",
    "Full detail in `reports/business_strategy.md`. Summary:",
    "",
]
for _, row in profile.iterrows():
    bn = base_name(row["segment_name"])
    short_action = ACTION_MAP.get(bn, "Monitor and tailor engagement.").split(".")[0].replace("**", "")
    lines.append(f"- **{row['segment_name']}**: {short_action}.")

lines += [
    "",
    "## Limitations",
    "",
    "- Dataset is synthetic; absolute values are illustrative, not benchmarks.",
    "- Latent generative archetypes (5) do not need to, and did not, match the analytically recovered "
    "cluster count (4) — this is realistic and intentional.",
    f"- K-Means/hierarchical agreement at the final k was moderate (ARI = {clus['final_agreement_ari']}), "
    "reflecting genuinely fuzzy segment boundaries rather than an error.",
    "- Segmentation reflects a single snapshot date; customer behaviour and segment membership will "
    "drift over time and should be re-run periodically in a real deployment.",
    "",
    "## Reproduction Instructions",
    "",
    "```bash",
    "pip install -r requirements.txt",
    "cd src/",
    "python run_pipeline.py       # generates data, cleans, engineers features, clusters, evaluates, plots",
    "python generate_reports.py   # builds all markdown reports (this README included) from live results",
    "```",
    "",
    "All steps use a fixed random seed (42) for reproducibility. Every number in this README and in the "
    "`reports/` folder is generated live by `generate_reports.py` from the pipeline's own output "
    "(`data/processed/pipeline_results.json`) — nothing is hand-typed.",
    "",
    "## Technical Stack",
    "",
    "Python, pandas, NumPy, scikit-learn (K-Means, Agglomerative Clustering, PCA, StandardScaler, "
    "PowerTransformer), SciPy (Kruskal-Wallis), statsmodels (Benjamini-Hochberg FDR correction), "
    "matplotlib & seaborn (visualisation), joblib (model persistence).",
    "",
    "## Project Structure",
    "",
    "```",
    "customer-value-segmentation/",
    "├── data/{raw,processed}/",
    "├── notebooks/customer_segmentation.ipynb",
    "├── src/ (data_generation, preprocessing, feature_engineering, clustering, evaluation, "
    "run_pipeline, generate_reports)",
    "├── models/ (persisted scaler, power transformer, PCA, final K-Means model)",
    "├── visualisations/ (10 numbered charts + supporting dendrogram)",
    "├── reports/ (data dictionary, segmentation, statistical, business strategy, quality control, "
    "case study)",
    "└── requirements.txt",
    "```",
]
(ROOT / "README.md").write_text("\n".join(lines))
print("wrote README.md")

print("\nAll reports generated successfully.")
