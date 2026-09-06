"""
run_pipeline.py
================
End-to-end orchestration for the Customer Value Segmentation project.

Runs: data generation -> cleaning -> feature engineering -> clustering
model selection -> final segmentation -> statistical validation ->
visualisations -> markdown reports.

Usage:
    cd src/
    python run_pipeline.py

All outputs are written under the project root (data/, models/,
visualisations/, reports/). Every number quoted in the generated reports and
README is computed live by this script — nothing is hard-coded.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats

import data_generation as dg
import preprocessing as pp
import feature_engineering as fe
import clustering as cl
import evaluation as ev

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
VIZ_DIR = ROOT / "visualisations"
REPORTS_DIR = ROOT / "reports"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, VIZ_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DATE = "2025-01-31"
FINAL_K = 4
ALPHA = 0.05

# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="deep")
PALETTE = sns.color_palette("Set2", 10)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 11,
    "axes.titleweight": "bold", "axes.titlesize": 13,
})


def save_fig(fig, name):
    path = VIZ_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


# =======================================================================
# STEP 1 — DATA GENERATION
# =======================================================================
def step_generate_data():
    print("\n[1/8] Generating synthetic dataset...")
    customers, transactions = dg.main(output_dir=str(DATA_RAW))
    return customers, transactions


# =======================================================================
# STEP 2 — VALIDATION & CLEANING
# =======================================================================
def step_clean_data():
    print("\n[2/8] Validating & cleaning data...")
    customers, transactions = pp.load_raw(DATA_RAW / "customers.csv", DATA_RAW / "transactions.csv")
    raw_customer_count = len(customers)
    raw_tx_count = len(transactions)
    customers_clean, transactions_clean, log = pp.validate_and_clean(customers, transactions)
    log["raw_customer_count"] = int(raw_customer_count)
    log["raw_transaction_count"] = int(raw_tx_count)
    print(f"  customers: {raw_customer_count:,} raw -> {len(customers_clean):,} clean")
    print(f"  transactions: {raw_tx_count:,} raw -> {len(transactions_clean):,} clean")
    return customers_clean, transactions_clean, log


# =======================================================================
# STEP 3 — FEATURE ENGINEERING
# =======================================================================
def step_engineer_features(customers_clean, transactions_clean):
    print("\n[3/8] Engineering RFM & behavioural features...")
    features = fe.build_customer_features(customers_clean, transactions_clean, snapshot_date=SNAPSHOT_DATE)
    features.to_csv(DATA_PROCESSED / "customer_features.csv", index=False)
    print(f"  {features.shape[1]} features for {features.shape[0]:,} customers -> data/processed/customer_features.csv")
    return features


# =======================================================================
# STEP 4 — EXPLORATORY ANALYSIS + VISUALS 1 & 2
# =======================================================================
def step_eda(features):
    print("\n[4/8] Exploratory analysis...")
    rfm_cols = ["recency_days", "frequency", "monetary_total"]

    skew_summary = features[rfm_cols + ["avg_order_value", "tenure_months"]].skew().round(2).to_dict()

    # Visual 1: RFM distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col, title, color in zip(
        axes, rfm_cols,
        ["Recency (days since last order)", "Frequency (orders in window)", "Monetary (total spend, £)"],
        PALETTE
    ):
        sns.histplot(features[col], bins=50, ax=ax, color=color, kde=True)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("Figure 1 — RFM Distributions (raw scale)", y=1.03, fontsize=14)
    save_fig(fig, "01_rfm_distributions")

    # Visual 2: correlation matrix
    corr_features = [
        "recency_days", "frequency", "monetary_total", "avg_order_value",
        "median_order_value", "order_value_std", "tenure_months",
        "distinct_categories_purchased", "return_rate", "orders_per_tenure_month",
        "spend_per_tenure_month", "engagement_score", "support_tickets_12m",
        "avg_satisfaction_score",
    ]
    corr = features[corr_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Pearson correlation"}, annot_kws={"size": 7})
    ax.set_title("Figure 2 — Correlation Matrix of Candidate Features")
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, "02_correlation_matrix")

    # Redundancy check: pairs with |r| > 0.8
    redundant_pairs = []
    for i, c1 in enumerate(corr_features):
        for c2 in corr_features[i + 1:]:
            r = corr.loc[c1, c2]
            if abs(r) > 0.8:
                redundant_pairs.append((c1, c2, round(float(r), 3)))

    eda_summary = {
        "skewness": {k: float(v) for k, v in skew_summary.items()},
        "redundant_pairs_r_gt_0_8": redundant_pairs,
        "n_customers_never_purchased": int(features["never_purchased_flag"].sum()),
        "pct_never_purchased": round(100 * features["never_purchased_flag"].mean(), 2),
    }
    return eda_summary


# =======================================================================
# STEP 5 — CLUSTERING (model selection + final fit)
# =======================================================================
def step_clustering(features):
    print("\n[5/8] Preparing features & selecting cluster count...")
    X, prep_meta = cl.prepare_clustering_matrix(features)

    k_range = list(range(2, 9))
    inertias = cl.elbow_analysis(X, k_range)
    silhouettes = cl.silhouette_analysis(X, k_range)

    # Visual 3: elbow curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(inertias.keys()), list(inertias.values()), marker="o", color=PALETTE[0], linewidth=2)
    ax.axvline(FINAL_K, color=PALETTE[3], linestyle="--", label=f"Selected k = {FINAL_K}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Figure 3 — Elbow Method for Optimal k")
    ax.legend()
    save_fig(fig, "03_elbow_curve")

    # Visual 4: silhouette scores across k
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(silhouettes.keys()), list(silhouettes.values()), marker="o", color=PALETTE[1], linewidth=2)
    ax.axvline(FINAL_K, color=PALETTE[3], linestyle="--", label=f"Selected k = {FINAL_K}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score")
    ax.set_title("Figure 4 — Silhouette Score by k")
    ax.legend()
    save_fig(fig, "04_silhouette_by_k")

    # Stability + algorithm comparison across a few candidate k values
    candidate_ks = [2, 3, FINAL_K, 5]
    model_selection_table = []
    for k in candidate_ks:
        stability = cl.stability_analysis(X, k)
        comparison, km_labels, agg_labels = cl.compare_algorithms(X, k)
        model_selection_table.append({
            "k": k,
            "inertia": round(float(inertias[k]), 1),
            "kmeans_silhouette": round(float(comparison["kmeans"]["silhouette"]), 3),
            "hierarchical_silhouette": round(float(comparison["hierarchical"]["silhouette"]), 3),
            "kmeans_davies_bouldin": round(float(comparison["kmeans"]["davies_bouldin"]), 3),
            "kmeans_vs_hier_agreement_ari": round(float(comparison["agreement_ari"]), 3),
            "bootstrap_stability_ari": round(float(stability["mean_ari"]), 3),
        })
    model_selection_df = pd.DataFrame(model_selection_table)

    # Final fit at FINAL_K with both algorithms for the report / dendrogram
    km_final, km_labels = cl.fit_kmeans(X, FINAL_K)
    agg_final, agg_labels = cl.fit_hierarchical(X, FINAL_K)
    final_comparison, _, _ = cl.compare_algorithms(X, FINAL_K)[0], None, None
    final_comparison = cl.compare_algorithms(X, FINAL_K)[0]

    # Visual: dendrogram sample (for hierarchical comparison, not one of the
    # numbered 10 core visuals but valuable supporting evidence)
    sample_idx = np.random.default_rng(42).choice(len(X), size=min(200, len(X)), replace=False)
    Z = linkage(X.iloc[sample_idx], method="ward")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, color_threshold=0)
    ax.set_title("Supporting Figure — Hierarchical Clustering Dendrogram (200-customer sample, truncated)")
    ax.set_xlabel("Sample cluster leaves")
    ax.set_ylabel("Ward linkage distance")
    save_fig(fig, "00_dendrogram_sample")

    features = features.copy()
    features["segment"] = km_labels
    features["segment_hierarchical"] = agg_labels

    # PCA for visualisation
    pca, components, explained, loadings = cl.run_pca(X, n_components=2)
    features["pca_1"] = components[:, 0]
    features["pca_2"] = components[:, 1]

    joblib.dump(km_final, MODELS_DIR / "kmeans_final.joblib")
    joblib.dump(prep_meta["scaler"], MODELS_DIR / "scaler.joblib")
    joblib.dump(prep_meta["power_transformer"], MODELS_DIR / "power_transformer.joblib")
    joblib.dump(pca, MODELS_DIR / "pca.joblib")
    print(f"  models saved to {MODELS_DIR.relative_to(ROOT)}/")

    clustering_summary = {
        "features_used": prep_meta["features"],
        "k_range_tested": k_range,
        "inertias": {int(k): round(float(v), 1) for k, v in inertias.items()},
        "silhouettes": {int(k): round(float(v), 4) for k, v in silhouettes.items()},
        "model_selection_table": model_selection_df.to_dict(orient="records"),
        "final_k": FINAL_K,
        "final_kmeans_silhouette": round(float(final_comparison["kmeans"]["silhouette"]), 4),
        "final_hierarchical_silhouette": round(float(final_comparison["hierarchical"]["silhouette"]), 4),
        "final_kmeans_davies_bouldin": round(float(final_comparison["kmeans"]["davies_bouldin"]), 4),
        "final_kmeans_calinski_harabasz": round(float(final_comparison["kmeans"]["calinski_harabasz"]), 2),
        "final_agreement_ari": round(float(final_comparison["agreement_ari"]), 4),
        "final_stability_ari": round(float(cl.stability_analysis(X, FINAL_K)["mean_ari"]), 4),
        "pca_explained_variance": [round(float(v), 4) for v in explained],
        "pca_loadings": loadings.round(3).to_dict(),
    }
    return features, X, clustering_summary


# =======================================================================
# STEP 6 — SEGMENT PROFILING, NAMING, VISUALS 5-9
# =======================================================================
def step_profiling(features, clustering_summary):
    print("\n[6/8] Profiling & naming segments...")
    profile = ev.profile_segments(features, "segment")
    names = ev.name_segments(profile)
    profile = profile.merge(names, on="segment")
    features = features.merge(names, on="segment", how="left")

    order = profile.sort_values("avg_monetary_total", ascending=False)["segment"].tolist()
    seg_color_map = {seg: PALETTE[i] for i, seg in enumerate(order)}

    def seg_label(row):
        return f"{row['segment_name']} (n={row['customer_count']:,})"

    profile["display_label"] = profile.apply(seg_label, axis=1)

    # Visual 5: PCA cluster visualisation
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for seg in order:
        sub = features[features["segment"] == seg]
        name = profile.loc[profile["segment"] == seg, "segment_name"].values[0]
        ax.scatter(sub["pca_1"], sub["pca_2"], s=10, alpha=0.5, color=seg_color_map[seg], label=name)
    var1, var2 = clustering_summary["pca_explained_variance"][:2]
    ax.set_xlabel(f"PC1 ({var1*100:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({var2*100:.1f}% variance explained)")
    ax.set_title("Figure 5 — Customer Segments in PCA Space")
    ax.legend(markerscale=2, fontsize=9)
    save_fig(fig, "05_pca_clusters")

    # Visual 6: segment size
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = profile.sort_values("customer_count", ascending=False)
    bars = ax.bar(plot_df["segment_name"], plot_df["customer_count"],
                   color=[seg_color_map[s] for s in plot_df["segment"]])
    for b, pct in zip(bars, plot_df["pct_of_customers"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of customers")
    ax.set_title("Figure 6 — Segment Size")
    plt.xticks(rotation=20, ha="right")
    save_fig(fig, "06_segment_size")

    # Visual 7: segment value (share of total monetary value)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = profile.sort_values("pct_of_total_monetary", ascending=False)
    bars = ax.bar(plot_df["segment_name"], plot_df["pct_of_total_monetary"],
                   color=[seg_color_map[s] for s in plot_df["segment"]])
    for b, v in zip(bars, plot_df["pct_of_total_monetary"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("% of total customer spend")
    ax.set_title("Figure 7 — Segment Share of Total Monetary Value")
    plt.xticks(rotation=20, ha="right")
    save_fig(fig, "07_segment_value")

    # Visual 8: segment behavioural profiles (radar-style via normalised bar grid)
    behav_metrics = ["avg_recency_days", "avg_frequency", "avg_monetary_total",
                      "avg_tenure_months", "avg_distinct_categories_purchased", "avg_return_rate"]
    norm = profile[["segment"] + behav_metrics].copy()
    for m in behav_metrics:
        rng_ = norm[m].max() - norm[m].min()
        norm[m] = (norm[m] - norm[m].min()) / rng_ if rng_ > 0 else 0.0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(behav_metrics))
    width = 0.8 / len(order)
    for i, seg in enumerate(order):
        row = norm[norm["segment"] == seg][behav_metrics].values.flatten()
        name = profile.loc[profile["segment"] == seg, "segment_name"].values[0]
        ax.bar(x + i * width, row, width=width, color=seg_color_map[seg], label=name)
    ax.set_xticks(x + width * (len(order) - 1) / 2)
    ax.set_xticklabels([m.replace("avg_", "").replace("_", " ") for m in behav_metrics], rotation=20, ha="right")
    ax.set_ylabel("Min-max normalised value (0-1, within metric)")
    ax.set_title("Figure 8 — Segment Behavioural Profiles (normalised)")
    ax.legend(fontsize=8)
    save_fig(fig, "08_behavioural_profiles")

    # Visual 9: segment comparison (RFM small multiples box plots)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in zip(
        axes, ["recency_days", "frequency", "monetary_total"],
        ["Recency (days)", "Frequency (orders)", "Monetary (£)"]
    ):
        plot_data = features.copy()
        plot_data["segment_name"] = plot_data["segment"].map(
            dict(zip(profile["segment"], profile["segment_name"]))
        )
        sns.boxplot(data=plot_data, x="segment_name", y=col, ax=ax,
                    order=[profile.loc[profile["segment"] == s, "segment_name"].values[0] for s in order],
                    palette=[seg_color_map[s] for s in order], showfliers=False)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Figure 9 — RFM Comparison Across Segments (outliers hidden for readability)", y=1.04)
    save_fig(fig, "09_segment_comparison")

    features.to_csv(DATA_PROCESSED / "customer_segments.csv", index=False)
    profile.to_csv(DATA_PROCESSED / "segment_profile.csv", index=False)
    print(f"  segment-level data saved to {DATA_PROCESSED.relative_to(ROOT)}/")

    return features, profile, order, seg_color_map


# =======================================================================
# STEP 7 — STATISTICAL VALIDATION
# =======================================================================
def step_statistics(features):
    print("\n[7/8] Running statistical validation...")
    stats_df = ev.statistical_validation(features, "segment", alpha=ALPHA)
    stats_df.to_csv(DATA_PROCESSED / "statistical_validation.csv", index=False)
    return stats_df


# =======================================================================
# STEP 8 — EXECUTIVE DASHBOARD (Visual 10)
# =======================================================================
def step_dashboard(features, profile, order, seg_color_map, clustering_summary):
    print("\n[8/8] Building executive dashboard...")

    total_customers = len(features)
    n_segments = len(profile)
    largest = profile.loc[profile["customer_count"].idxmax()]
    highest_value = profile.loc[profile["pct_of_total_monetary"].idxmax()]
    # "At-risk" segment = named at-risk if present, else the segment with worst recency among value>average
    at_risk_candidates = profile[profile["segment_name"].str.contains("At-Risk", case=False)]
    at_risk = at_risk_candidates.iloc[0] if len(at_risk_candidates) else profile.loc[profile["avg_recency_days"].idxmax()]

    top_20pct_customers_n = max(1, int(0.2 * total_customers))
    monetary_sorted = features.sort_values("monetary_total", ascending=False)
    top20_share = 100 * monetary_sorted.head(top_20pct_customers_n)["monetary_total"].sum() / features["monetary_total"].sum()

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.4)

    kpis = [
        ("Total Customers", f"{total_customers:,}"),
        ("Segments Identified", f"{n_segments}"),
        ("Largest Segment", f"{largest['segment_name']}\n({largest['pct_of_customers']:.1f}%)"),
        ("Highest-Value Segment", f"{highest_value['segment_name']}\n({highest_value['pct_of_total_monetary']:.1f}% of spend)"),
        ("At-Risk Segment", f"{at_risk['segment_name']}\n({at_risk['customer_count']:,} customers)"),
        ("Top 20% Customers Drive", f"{top20_share:.1f}% of spend"),
    ]
    for i, (label, value) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i % 4]) if i < 4 else fig.add_subplot(gs[1, i - 4])
        ax.axis("off")
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=15 if len(value) < 20 else 12,
                fontweight="bold", color="#1f3d5c", wrap=True)
        ax.text(0.5, 0.15, label, ha="center", va="center", fontsize=10.5, color="#555555")
        ax.add_patch(plt.Rectangle((0.03, 0.03), 0.94, 0.94, fill=False, edgecolor="#cccccc", lw=1.2,
                                    transform=ax.transAxes))

    ax_dist = fig.add_subplot(gs[1, 2:4])
    plot_df = profile.sort_values("customer_count", ascending=False)
    ax_dist.pie(plot_df["customer_count"], labels=plot_df["segment_name"], autopct="%1.1f%%",
                colors=[seg_color_map[s] for s in plot_df["segment"]], textprops={"fontsize": 8.5})
    ax_dist.set_title("Segment Distribution", fontsize=12, fontweight="bold")

    ax_val = fig.add_subplot(gs[2, 0:2])
    plot_df2 = profile.sort_values("pct_of_total_monetary", ascending=False)
    ax_val.barh(plot_df2["segment_name"], plot_df2["pct_of_total_monetary"],
                color=[seg_color_map[s] for s in plot_df2["segment"]])
    ax_val.set_xlabel("% of total spend")
    ax_val.set_title("Value Concentration by Segment", fontsize=12, fontweight="bold")
    ax_val.invert_yaxis()

    ax_action = fig.add_subplot(gs[2, 2:4])
    ax_action.axis("off")
    action_map = {
        "High-Value Loyal": "Retain & reward: loyalty perks, early access, premium tiers",
        "At-Risk High Value": "Win back: personal outreach, service recovery, targeted incentives",
        "At-Risk / Lapsing": "Re-engage: win-back campaigns, satisfaction check-ins",
        "Emerging Customers": "Grow: cross-sell, category expansion, loyalty enrolment",
        "New / Low Engagement": "Onboard: welcome journeys, first-purchase incentives",
        "Dormant Long-Tenure": "Reactivate or gracefully sunset: relevance-testing campaigns",
        "Steady Mid-Value": "Nurture: consistent engagement, incremental upsell",
    }
    text_lines = ["Recommended Action per Segment", ""]
    for _, row in profile.sort_values("pct_of_total_monetary", ascending=False).iterrows():
        base_name = row["segment_name"].split(" (Segment")[0]
        action = action_map.get(base_name, "Monitor & tailor engagement")
        text_lines.append(f"• {row['segment_name']}: {action}")
    ax_action.text(0, 1, "\n".join(text_lines), va="top", ha="left", fontsize=9.3, wrap=True,
                   transform=ax_action.transAxes)

    fig.suptitle("Figure 10 — Customer Value Segmentation: Executive Dashboard", fontsize=17, fontweight="bold", y=0.98)
    save_fig(fig, "10_executive_dashboard")

    dashboard_summary = {
        "total_customers": total_customers,
        "n_segments": n_segments,
        "largest_segment": largest["segment_name"],
        "largest_segment_pct": float(largest["pct_of_customers"]),
        "highest_value_segment": highest_value["segment_name"],
        "highest_value_segment_pct_spend": float(highest_value["pct_of_total_monetary"]),
        "at_risk_segment": at_risk["segment_name"],
        "at_risk_segment_count": int(at_risk["customer_count"]),
        "top20pct_customers_spend_share": round(float(top20_share), 2),
    }
    return dashboard_summary


def main():
    step_generate_data()
    customers_clean, transactions_clean, clean_log = step_clean_data()
    features = step_engineer_features(customers_clean, transactions_clean)
    eda_summary = step_eda(features)
    features, X, clustering_summary = step_clustering(features)
    features, profile, order, seg_color_map = step_profiling(features, clustering_summary)
    stats_df = step_statistics(features)
    dashboard_summary = step_dashboard(features, profile, order, seg_color_map, clustering_summary)

    # Persist a single JSON with every computed number, consumed by the
    # report-generation script so reports never hard-code results.
    all_results = {
        "clean_log": clean_log,
        "eda_summary": eda_summary,
        "clustering_summary": clustering_summary,
        "profile": profile.to_dict(orient="records"),
        "stats": stats_df.to_dict(orient="records"),
        "dashboard": dashboard_summary,
    }
    with open(DATA_PROCESSED / "pipeline_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nPipeline complete. All results saved to data/processed/pipeline_results.json")
    return all_results


if __name__ == "__main__":
    main()
