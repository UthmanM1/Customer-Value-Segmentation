"""Builds customer_segmentation.ipynb as a narrated, executable walkthrough
of the full pipeline (mirrors src/run_pipeline.py but with explanatory
markdown between steps, for a portfolio-reviewer audience)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Customer Value Segmentation

**Portfolio project — synthetic data.** This notebook walks through an end-to-end unsupervised
customer segmentation analysis for a fictional UK subscription/retail business: data generation,
cleaning, RFM & behavioural feature engineering, clustering model selection, segment profiling,
statistical validation, and business strategy.

All data is synthetically generated (`src/data_generation.py`) and does not represent a real
company. The full pipeline (identical logic to this notebook) can also be run non-interactively via
`src/run_pipeline.py` + `src/generate_reports.py`.
""")

code("""import sys
sys.path.insert(0, '../src')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams.update({'figure.dpi': 100})

import data_generation as dg
import preprocessing as pp
import feature_engineering as fe
import clustering as cl
import evaluation as ev

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 160)
""")

md("""## 1. Business Scenario & Synthetic Data Generation

The company has 10,000+ customers but treats them as one group. We generate a synthetic transaction
history (customers + 12+ months of orders) with **realistic data-quality issues** and **overlapping,
noisy behavioural archetypes** — deliberately avoiding obviously-separated clusters, so the
segmentation that follows requires genuine analysis rather than reading off a pre-baked label.
""")

code("""customers_raw, transactions_raw = dg.main(output_dir='../data/raw')
print(customers_raw.shape, transactions_raw.shape)
customers_raw.head()
""")

code("""transactions_raw.head()
""")

md("""## 2. Data Validation & Cleaning

We check for duplicates, missing values, invalid values (negative order values), and outliers, and
apply **documented, justified** treatment decisions for each (see `reports/quality_control_report.md`
for the full rationale — e.g. why outliers are winsorised rather than dropped).
""")

code("""customers_clean, transactions_clean, clean_log = pp.validate_and_clean(customers_raw, transactions_raw)
for k, v in clean_log.items():
    print(f"{k}: {v}")
""")

md("""## 3. Feature Engineering: RFM + Behavioural Features

We aggregate transactions to customer level and construct **Recency, Frequency, Monetary (RFM)**
variables, plus tenure, category breadth, return rate, and tenure-normalised activity measures.
Field definitions: `reports/data_dictionary.md`.
""")

code("""features = fe.build_customer_features(customers_clean, transactions_clean, snapshot_date='2025-01-31')
print(features.shape)
features[['customer_id','recency_days','frequency','monetary_total','avg_order_value','tenure_months']].describe()
""")

md("""## 4. Exploratory Analysis

### 4.1 RFM Distributions
RFM variables are typically right-skewed in retail data — a small number of very active/high-spend
customers create a long tail. We check this explicitly rather than assuming it.
""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title in zip(axes, ['recency_days','frequency','monetary_total'],
                          ['Recency (days)','Frequency (orders)','Monetary (£, total spend)']):
    sns.histplot(features[col], bins=50, ax=ax, kde=True)
    ax.set_title(title)
plt.tight_layout()
plt.show()

print('Skewness:')
print(features[['recency_days','frequency','monetary_total','avg_order_value','tenure_months']].skew().round(2))
""")

md("""### 4.2 Correlation & Redundancy

We check for highly correlated (redundant) features before clustering. Correlated pairs are noted;
some are kept anyway if they carry *distinct business meaning* (e.g. frequency vs monetary value —
a customer who buys often at low value differs strategically from one who buys rarely at high
value), with the reasoning made explicit rather than dropping features mechanically.
""")

code("""corr_features = ['recency_days','frequency','monetary_total','avg_order_value','median_order_value',
                  'order_value_std','tenure_months','distinct_categories_purchased','return_rate',
                  'orders_per_tenure_month','spend_per_tenure_month','engagement_score',
                  'support_tickets_12m','avg_satisfaction_score']
corr = features[corr_features].corr()

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, annot_kws={'size':7})
plt.xticks(rotation=45, ha='right')
plt.title('Correlation Matrix of Candidate Features')
plt.show()

redundant = [(a,b,round(float(corr.loc[a,b]),3)) for i,a in enumerate(corr_features)
             for b in corr_features[i+1:] if abs(corr.loc[a,b]) > 0.8]
print('Redundant pairs (|r| > 0.8):', redundant)
""")

md("""## 5. Preparing Features for Clustering

Skewed features are power-transformed (**Yeo-Johnson**, which — unlike log1p — handles zero and
negative values while correcting skew), then **all features are standardised** to zero mean / unit
variance. Standardisation matters because distance-based clustering (K-Means, Ward-linkage
hierarchical) is dominated by whichever feature has the largest raw scale — without it,
`monetary_total` (scale: £0–thousands) would swamp `return_rate` (scale: 0–1) regardless of true
business importance.
""")

code("""X, prep_meta = cl.prepare_clustering_matrix(features)
print('Clustering feature matrix:', X.shape)
print('\\nFeatures used:', prep_meta['features'])
X.describe().round(2)
""")

md("""## 6. Cluster Count Selection: Elbow, Silhouette & Stability

We compare **K-Means** and **hierarchical (Ward-linkage) clustering** across a range of k, using the
elbow method (inertia), mean silhouette score, Davies-Bouldin index, and **bootstrap stability**
(mean Adjusted Rand Index across 10 resampled refits — a high value means the partition is
reproducible, not an artefact of one particular sample).
""")

code("""k_range = range(2, 9)
inertias = cl.elbow_analysis(X, k_range)
silhouettes = cl.silhouette_analysis(X, k_range)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(list(inertias.keys()), list(inertias.values()), marker='o')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].set_title('Elbow Method')
axes[1].plot(list(silhouettes.keys()), list(silhouettes.values()), marker='o', color='darkorange')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Mean silhouette'); axes[1].set_title('Silhouette Score by k')
plt.tight_layout(); plt.show()

print('Inertias:', {k: round(v,1) for k,v in inertias.items()})
print('Silhouettes:', {k: round(v,4) for k,v in silhouettes.items()})
""")

code("""model_selection = []
for k in [2, 3, 4, 5]:
    stability = cl.stability_analysis(X, k)
    comparison, km_labels, agg_labels = cl.compare_algorithms(X, k)
    model_selection.append({
        'k': k,
        'kmeans_silhouette': round(comparison['kmeans']['silhouette'], 3),
        'hierarchical_silhouette': round(comparison['hierarchical']['silhouette'], 3),
        'kmeans_davies_bouldin': round(comparison['kmeans']['davies_bouldin'], 3),
        'kmeans_vs_hier_agreement_ari': round(comparison['agreement_ari'], 3),
        'bootstrap_stability_ari': round(stability['mean_ari'], 3),
    })
pd.DataFrame(model_selection)
""")

md("""### Why k = 4, not the k with the highest silhouette score

**k = 2 has the highest raw silhouette score** — but inspecting that partition shows it almost
perfectly separates customers into "has purchased" vs "has never purchased." That's real structure,
but far too coarse to drive differentiated retention/marketing strategy on its own.

**k = 3** offers a slightly better silhouette/stability trade-off than k = 4, but it collapses two
behaviourally and strategically distinct groups — *newly acquired, low-activity customers* and
*long-tenured customers who have gone quiet* — into a single middle cluster. Those two groups need
opposite interventions (onboarding vs. win-back), so merging them would actively mislead strategy.

**k = 4** is the smallest k that cleanly separates those groups while keeping bootstrap stability
high and silhouette respectable. Beyond k = 4, the elbow curve flattens and additional clusters
mostly subdivide existing groups without new business-relevant distinctions.

We proceed with **k = 4**, fit with K-Means (chosen over hierarchical clustering at this k for its
higher silhouette score, lower Davies-Bouldin index, and better scalability).
""")

code("""FINAL_K = 4
km_final, km_labels = cl.fit_kmeans(X, FINAL_K)
agg_final, agg_labels = cl.fit_hierarchical(X, FINAL_K)

features['segment'] = km_labels
features['segment_hierarchical'] = agg_labels

final_comparison, _, _ = cl.compare_algorithms(X, FINAL_K)
print('Final K-Means silhouette:', round(final_comparison['kmeans']['silhouette'], 4))
print('Final Hierarchical silhouette:', round(final_comparison['hierarchical']['silhouette'], 4))
print('K-Means vs Hierarchical agreement (ARI):', round(final_comparison['agreement_ari'], 4))
print('Bootstrap stability (ARI):', round(cl.stability_analysis(X, FINAL_K)['mean_ari'], 4))
""")

md("""## 7. Dimensionality Reduction (PCA) for Visualisation

PCA is used **only to visualise** the already-fitted cluster solution in 2D — clustering itself was
performed on the full standardised feature space, not the PCA projection. The components are linear
combinations of every input feature, so we avoid over-interpreting PC1/PC2 as clean business
concepts like "value" or "engagement."
""")

code("""pca, components, explained, loadings = cl.run_pca(X, n_components=2)
features['pca_1'] = components[:, 0]
features['pca_2'] = components[:, 1]
print(f"Variance explained: PC1={explained[0]*100:.1f}%, PC2={explained[1]*100:.1f}% "
      f"(combined {sum(explained[:2])*100:.1f}%)")
loadings.round(3)
""")

code("""profile_preview = ev.profile_segments(features, 'segment')
names_preview = ev.name_segments(profile_preview)
seg_name_map = dict(zip(names_preview['segment'], names_preview['segment_name']))
features['segment_name'] = features['segment'].map(seg_name_map)

fig, ax = plt.subplots(figsize=(8,6.5))
for seg_name, sub in features.groupby('segment_name'):
    ax.scatter(sub['pca_1'], sub['pca_2'], s=10, alpha=0.5, label=seg_name)
ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% var)')
ax.set_title('Customer Segments in PCA Space')
ax.legend(markerscale=2)
plt.show()
""")

md("""## 8. Segment Profiling & Business-Friendly Naming

Segment names are **derived programmatically** from each segment's rank on value (monetary +
frequency), recency, and tenure relative to the *other segments in this run* — not hard-coded — so
they adapt automatically if re-clustering changes segment composition.
""")

code("""profile = ev.profile_segments(features, 'segment')
names = ev.name_segments(profile)
profile = profile.merge(names, on='segment')
profile = profile.sort_values('pct_of_total_monetary', ascending=False).reset_index(drop=True)
profile[['segment_name','customer_count','pct_of_customers','pct_of_total_monetary',
         'avg_recency_days','avg_frequency','avg_monetary_total','avg_tenure_months']]
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].bar(profile['segment_name'], profile['customer_count'])
axes[0].set_title('Segment Size'); axes[0].tick_params(axis='x', rotation=20)
axes[1].bar(profile['segment_name'], profile['pct_of_total_monetary'], color='darkorange')
axes[1].set_title('Segment Share of Total Spend (%)'); axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.show()
""")

md("""## 9. Statistical Validation

We use the **Kruskal-Wallis H-test** (non-parametric — appropriate given the non-normal, skewed RFM
distributions) to test whether each metric genuinely differs across segments, report **effect size
(epsilon-squared)** to judge practical (not just statistical) significance, and apply
**Benjamini-Hochberg FDR correction** since many metrics are tested simultaneously.
""")

code("""stats_results = ev.statistical_validation(features, 'segment')
stats_results[['metric','h_statistic','p_value','p_value_fdr_corrected',
               'significant_after_correction','effect_size_epsilon_sq','effect_size_label']]
""")

md("""Large effect sizes (e.g. `monetary_total`, `frequency`, `tenure_months`) confirm the segments
differ substantially in the variables the business cares about most. A few metrics
(`avg_satisfaction_score`, `engagement_score`, `support_tickets_12m`) show negligible effect sizes —
this is reported honestly rather than selectively omitted, since these variables were not part of the
clustering feature set and should not be assumed to distinguish segments.
""")

md("""## 10. Business Strategy Summary

Each segment's strategy is grounded directly in its computed profile (see
`reports/business_strategy.md` for full detail per segment):
""")

code("""strategy_map = {
    'High-Value Loyal': 'Retain & reward — loyalty perks, premium tiers, proactive relationship management.',
    'At-Risk High Value': 'Targeted win-back — personal outreach and service recovery.',
    'At-Risk / Lapsing': 'Re-engagement campaigns — win-back offers and satisfaction check-ins.',
    'Emerging Customers': 'Growth & cross-sell — category expansion, loyalty enrolment.',
    'New / Low Engagement': 'Onboarding — welcome journeys, first-purchase incentives.',
    'Dormant Long-Tenure': 'Reactivate or gracefully sunset — relevance-testing, then suppress spend.',
    'Steady Mid-Value': 'Nurture — consistent engagement, incremental upsell.',
}
for _, row in profile.iterrows():
    base = row['segment_name'].split(' (Segment')[0]
    print(f"{row['segment_name']} ({row['customer_count']:,} customers, "
          f"{row['pct_of_total_monetary']:.1f}% of spend): {strategy_map.get(base, 'Monitor and tailor.')}")
""")

md("""## 11. Executive Summary

- **Segments identified:** see table above.
- **Highest-value segment** drives the majority of total spend from a minority of customers —
  retention investment should be prioritised accordingly.
- **At-risk segment(s)** represent meaningful revenue that will be lost without timely, targeted
  intervention.
- Full executive dashboard: `visualisations/10_executive_dashboard.png`.
- Full written reports: `reports/` (data dictionary, segmentation report, statistical report,
  business strategy, quality control report, portfolio case study).

*This notebook mirrors `src/run_pipeline.py`; run that script (followed by `src/generate_reports.py`)
to regenerate all saved visualisations and markdown reports non-interactively.*
""")

nb['cells'] = cells
nbf.write(nb, 'customer_segmentation.ipynb')
print("Notebook written.")
