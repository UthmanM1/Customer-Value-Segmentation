# Portfolio Case Study: Customer Value Segmentation

*This is an independent portfolio project built to demonstrate professional-level customer analytics and unsupervised machine learning. It is based on a fully synthetic dataset and does not represent real client work or a real company.*

## The Business Problem

A UK subscription/retail business treats its entire customer base as one homogeneous group for marketing and retention purposes. Budget, messaging, and service levels are undifferentiated, meaning high-value customers receive no special retention treatment and low-engagement customers receive the same investment as loyal repeat buyers.

## Approach

1. Generated a realistic synthetic dataset (10,500 customers, 112,055 raw transactions over ~13 months) with intentional data-quality issues (duplicates, missing values, inconsistent formatting, outliers) to mirror a real operational extract.
2. Validated and cleaned the data with documented, justified treatment decisions for every issue found (see `quality_control_report.md`).
3. Engineered RFM (Recency, Frequency, Monetary) variables plus behavioural features (category breadth, return rate, tenure-normalised activity, engagement score).
4. Explored distributions, skewness, correlations and redundancy; applied a Yeo-Johnson transform and standardisation to prepare features for distance-based clustering.
5. Compared K-Means and hierarchical clustering across k = 2-8, using the elbow method, silhouette analysis, Davies-Bouldin index, and bootstrap stability to select **k = 4** — with an explicit written justification for not simply picking the highest silhouette score.
6. Statistically validated segment differences with Kruskal-Wallis tests, FDR correction, and effect sizes, rather than relying on visual inspection alone.
7. Profiled and named each segment based on its actual computed characteristics, and derived a specific, evidence-backed strategy for each.

## Key Results

- **4 segments** identified from 10,500 customers.
- **High-Value Loyal** is the largest segment (36.34% of customers).
- **High-Value Loyal** drives **83.3%** of total customer spend.
- **At-Risk / Lapsing** (2,992 customers) represents meaningful revenue at risk without targeted intervention.
- The top 20% of customers by value drive **63.01%** of total spend — a Pareto-style concentration that justifies segment-specific retention investment over undifferentiated marketing.

## Skills Demonstrated

- Synthetic data generation with realistic noise and quality issues
- Data validation, cleaning, and documented treatment decisions
- Feature engineering (RFM + behavioural) from transaction-level data
- Exploratory data analysis: skewness, correlation, redundancy
- Unsupervised learning: K-Means, hierarchical clustering, cluster-count selection methodology
- Dimensionality reduction (PCA) for visualisation, with correct caveats
- Non-parametric statistical hypothesis testing with multiple-testing correction and effect sizes
- Business translation: segment naming and strategy grounded in data, not assumption
- Reproducible pipeline engineering (fixed seeds, modular `src/`, single orchestration script)

## Reproduction

```bash
pip install -r requirements.txt
cd src/
python run_pipeline.py
python generate_reports.py
```