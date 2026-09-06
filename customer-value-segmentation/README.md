# Customer Value Segmentation

**An independent data science portfolio project** demonstrating customer analytics and unsupervised machine learning on a synthetic UK subscription/retail dataset.

> ⚠️ **This is a portfolio project, not real client work.** All data is synthetically generated (see `src/data_generation.py`) and does not represent any real company or customers.

## Business Challenge

A retail/subscription business currently treats its customer base as one undifferentiated group. This project identifies meaningful customer segments from transaction data and develops data-backed strategies for each.

## Objective

Segment customers using RFM (Recency, Frequency, Monetary) and behavioural features via unsupervised clustering, validate the segments statistically, and translate them into actionable business strategy.

## Dataset (Synthetic)

- **10,532** synthetic customers, **112,055** synthetic transactions across ~13 months (Jan 2024 – Jan 2025)
- 10 product categories, 10 UK regions, 6 acquisition channels
- Deliberately includes realistic data-quality issues: missing values, duplicate records, inconsistent categorical formatting, and outliers
- Customer behaviour is generated from 5 overlapping latent archetypes with substantial noise, so that no raw feature cleanly separates customers — genuine analysis is required to recover meaningful structure

## Data Preparation

- 32 duplicate customer rows and 889 duplicate transaction rows removed
- Missing values handled per-field (median imputation with indicator flags for numeric fields; explicit 'Unknown' category for missing region; rows dropped only for missing monetary values, which cannot be safely imputed)
- 8 negative order values corrected (sign errors)
- 1526 extreme outliers identified (3×IQR) and winsorised rather than dropped

Full detail: `reports/quality_control_report.md`.

## RFM Methodology

Recency, Frequency and Monetary value were computed per customer as of a fixed snapshot date (2025-01-31), alongside behavioural features (category breadth, return rate, tenure-normalised activity, engagement score). See `reports/data_dictionary.md` for full field definitions.

## Clustering Methodology

K-Means and Ward-linkage hierarchical clustering were compared across k = 2–8 using the elbow method, silhouette score, Davies-Bouldin index, and bootstrap stability (Adjusted Rand Index across resamples). Features were Yeo-Johnson power-transformed (for skew) and standardised before clustering.

## Cluster Selection

**k = 4** was selected. This was *not* the k with the single highest silhouette score (k=2, silhouette 0.3251, scored higher but only separated 'ever purchased' from 'never purchased' — too coarse to act on). k=4 was chosen because it is the smallest k that separates behaviourally and strategically distinct groups (in particular, separating newly-acquired low-activity customers from long-tenured customers who have gone quiet) while maintaining high bootstrap stability (mean ARI = 0.9778) and a respectable silhouette score (0.2471). Full reasoning: `reports/segmentation_report.md` §5.

## Segment Profiles

| Segment | Customers | % of Base | % of Spend | Avg Recency (days) | Avg Frequency | Avg Monetary |
|---|---|---|---|---|---|---|
| High-Value Loyal | 3,816 | 36.34% | 83.3% | 29.9 | 21.75 | £1,347.58 |
| Emerging Customers | 2,228 | 21.22% | 8.4% | 27.0 | 5.89 | £232.86 |
| At-Risk / Lapsing | 2,992 | 28.5% | 8.23% | 127.4 | 4.75 | £169.77 |
| New / Low Engagement | 1,464 | 13.94% | 0.07% | 105.1 | 0.21 | £2.95 |

## Key Insights

- **High-Value Loyal** (36.34% of customers) drives **83.3%** of total spend — retention here matters more than acquisition anywhere else in the base.
- **At-Risk / Lapsing** (2,992 customers) shows meaningfully worse recency despite historically reasonable purchase activity — a clear win-back target before these customers churn entirely.
- The top 20% of customers by monetary value account for **63.01%** of total spend, confirming a Pareto-style value concentration.
- 11 of 13 profiled metrics differ significantly across segments after multiple-testing correction, with several showing large effect sizes (not just statistical significance) — see `reports/statistical_report.md`.

## Business Strategies

Full detail in `reports/business_strategy.md`. Summary:

- **High-Value Loyal**: Retain & reward.
- **Emerging Customers**: Growth & cross-sell.
- **At-Risk / Lapsing**: Re-engagement campaigns.
- **New / Low Engagement**: Onboarding.

## Limitations

- Dataset is synthetic; absolute values are illustrative, not benchmarks.
- Latent generative archetypes (5) do not need to, and did not, match the analytically recovered cluster count (4) — this is realistic and intentional.
- K-Means/hierarchical agreement at the final k was moderate (ARI = 0.4216), reflecting genuinely fuzzy segment boundaries rather than an error.
- Segmentation reflects a single snapshot date; customer behaviour and segment membership will drift over time and should be re-run periodically in a real deployment.

## Reproduction Instructions

```bash
pip install -r requirements.txt
cd src/
python run_pipeline.py       # generates data, cleans, engineers features, clusters, evaluates, plots
python generate_reports.py   # builds all markdown reports (this README included) from live results
```

All steps use a fixed random seed (42) for reproducibility. Every number in this README and in the `reports/` folder is generated live by `generate_reports.py` from the pipeline's own output (`data/processed/pipeline_results.json`) — nothing is hand-typed.

## Technical Stack

Python, pandas, NumPy, scikit-learn (K-Means, Agglomerative Clustering, PCA, StandardScaler, PowerTransformer), SciPy (Kruskal-Wallis), statsmodels (Benjamini-Hochberg FDR correction), matplotlib & seaborn (visualisation), joblib (model persistence).

## Project Structure

```
customer-value-segmentation/
├── data/{raw,processed}/
├── notebooks/customer_segmentation.ipynb
├── src/ (data_generation, preprocessing, feature_engineering, clustering, evaluation, run_pipeline, generate_reports)
├── models/ (persisted scaler, power transformer, PCA, final K-Means model)
├── visualisations/ (10 numbered charts + supporting dendrogram)
├── reports/ (data dictionary, segmentation, statistical, business strategy, quality control, case study)
└── requirements.txt
```