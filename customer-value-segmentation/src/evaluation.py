"""
evaluation.py
=============
Segment profiling, statistical validation of cluster differences, and
business-friendly segment naming logic.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROFILE_METRICS = [
    "recency_days", "frequency", "monetary_total", "avg_order_value",
    "median_order_value", "tenure_months", "distinct_categories_purchased",
    "return_rate", "engagement_score", "support_tickets_12m",
    "avg_satisfaction_score", "orders_per_tenure_month", "spend_per_tenure_month",
]


def profile_segments(df: pd.DataFrame, cluster_col: str):
    n_total = len(df)
    rows = []
    for seg, g in df.groupby(cluster_col):
        row = {
            "segment": seg,
            "customer_count": len(g),
            "pct_of_customers": round(100 * len(g) / n_total, 2),
            "pct_of_total_monetary": round(100 * g["monetary_total"].sum() / df["monetary_total"].sum(), 2)
            if df["monetary_total"].sum() > 0 else 0.0,
        }
        for m in PROFILE_METRICS:
            if m in g.columns:
                row[f"avg_{m}"] = round(g[m].mean(), 2)
        rows.append(row)
    profile = pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)
    return profile


def statistical_validation(df: pd.DataFrame, cluster_col: str, metrics=None, alpha=0.05):
    """Kruskal-Wallis test (non-parametric, robust to non-normal / skewed RFM
    distributions) for each metric across segments, with effect size
    (epsilon-squared) and Benjamini-Hochberg FDR correction for multiple
    comparisons across metrics.
    """
    metrics = metrics or PROFILE_METRICS
    metrics = [m for m in metrics if m in df.columns]

    results = []
    for m in metrics:
        groups = [g[m].dropna().values for _, g in df.groupby(cluster_col)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        h_stat, p_value = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        k = len(groups)
        # epsilon-squared effect size for Kruskal-Wallis
        epsilon_sq = (h_stat - k + 1) / (n - k) if (n - k) > 0 else np.nan
        results.append({
            "metric": m, "h_statistic": h_stat, "p_value": p_value,
            "effect_size_epsilon_sq": epsilon_sq, "n": n, "k_groups": k
        })

    res_df = pd.DataFrame(results)
    if len(res_df) > 0:
        reject, p_adj, _, _ = multipletests(res_df["p_value"], alpha=alpha, method="fdr_bh")
        res_df["p_value_fdr_corrected"] = p_adj
        res_df["significant_after_correction"] = reject

        def effect_label(e):
            if pd.isna(e):
                return "n/a"
            if e < 0.01:
                return "negligible"
            elif e < 0.06:
                return "small"
            elif e < 0.14:
                return "medium"
            else:
                return "large"

        res_df["effect_size_label"] = res_df["effect_size_epsilon_sq"].apply(effect_label)
        res_df = res_df.sort_values("effect_size_epsilon_sq", ascending=False).reset_index(drop=True)

    return res_df


def name_segments(profile: pd.DataFrame):
    """Assign a business-friendly name to each segment based on its actual
    RFM characteristics RELATIVE to the other segments in this run.

    Names are DERIVED from percentile rank thresholds across whatever number
    of segments the clustering produced (not hard-coded to a fixed k), so the
    logic adapts if re-running the pipeline changes segment composition. A
    combined value score (monetary + frequency, equally weighted after
    min-max scaling) drives the value tier; recency and tenure position
    distinguish loyal / emerging / at-risk / new / dormant behaviour.
    """
    p = profile.copy()
    n = len(p)

    def pct_rank(series, ascending):
        # 0 = "best" position on this dimension, 1 = "worst"; robust to ties.
        r = series.rank(ascending=ascending, method="average")
        return (r - 1) / (n - 1) if n > 1 else pd.Series(0.0, index=series.index)

    # Combine monetary and frequency standing via RANK (not raw min-max),
    # so one extreme segment doesn't compress the other segments' relative
    # positions on a skewed absolute scale.
    monetary_pct_rank = pct_rank(p["avg_monetary_total"], ascending=False)
    frequency_pct_rank = pct_rank(p["avg_frequency"], ascending=False)
    p["monetary_pct"] = 0.5 * monetary_pct_rank + 0.5 * frequency_pct_rank  # 0 = highest value tier, 1 = lowest
    p["recency_pct"] = pct_rank(p["avg_recency_days"], ascending=True)   # 0 = most recent
    p["tenure_pct"] = pct_rank(p["avg_tenure_months"], ascending=False)  # 0 = longest tenure

    names = []
    for _, row in p.iterrows():
        mp, rp, tp = row["monetary_pct"], row["recency_pct"], row["tenure_pct"]

        if mp <= 0.3 and rp <= 0.5:
            name = "High-Value Loyal"
        elif mp <= 0.3 and rp > 0.5:
            name = "At-Risk High Value"
        elif mp > 0.7 and tp > 0.5:
            name = "New / Low Engagement"
        elif mp > 0.7 and tp <= 0.5:
            name = "Dormant Long-Tenure"
        elif rp > 0.5 and tp <= 0.5:
            name = "At-Risk / Lapsing"
        elif rp <= 0.5:
            name = "Emerging Customers"
        else:
            name = "Steady Mid-Value"
        names.append(name)

    p["segment_name"] = names

    # Deduplicate names if two segments genuinely collide (append distinguishing detail)
    if p["segment_name"].duplicated().any():
        dup_mask = p["segment_name"].duplicated(keep=False)
        for idx in p[dup_mask].index:
            seg_id = p.loc[idx, "segment"]
            p.loc[idx, "segment_name"] = f"{p.loc[idx, 'segment_name']} (Segment {seg_id})"

    return p[["segment", "segment_name"]]


if __name__ == "__main__":
    import preprocessing as pp
    import feature_engineering as fe
    import clustering as cl

    customers, transactions = pp.load_raw("../data/raw/customers.csv", "../data/raw/transactions.csv")
    customers_clean, transactions_clean, log = pp.validate_and_clean(customers, transactions)
    features = fe.build_customer_features(customers_clean, transactions_clean, snapshot_date="2025-01-31")
    X, meta = cl.prepare_clustering_matrix(features)
    km, labels = cl.fit_kmeans(X, 4)
    features["segment"] = labels

    profile = profile_segments(features, "segment")
    print(profile)

    names = name_segments(profile)
    print(names)

    stats_res = statistical_validation(features, "segment")
    print(stats_res)
