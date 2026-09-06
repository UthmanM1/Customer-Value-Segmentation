"""
clustering.py
=============
Preprocessing-for-clustering (transform + scale), model comparison
(K-Means vs Agglomerative/Hierarchical clustering), cluster-count selection
(elbow + silhouette + stability), and PCA for visualisation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.utils import resample

RANDOM_SEED = 42

CLUSTERING_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "avg_order_value",
    "distinct_categories_purchased",
    "return_rate",
    "tenure_months",
    "orders_per_tenure_month",
    "spend_per_tenure_month",
    "engagement_score",
]

# Features that are typically right-skewed and benefit from a power transform
SKEWED_FEATURES = [
    "recency_days", "frequency", "monetary_total", "avg_order_value",
    "orders_per_tenure_month", "spend_per_tenure_month",
]


def prepare_clustering_matrix(df: pd.DataFrame, features=None):
    """Select features, apply a Yeo-Johnson power transform to skewed
    variables (handles zeros, unlike log1p, and corrects both left/right
    skew), then standardise everything to zero mean / unit variance so that
    no single feature dominates Euclidean distance calculations.
    """
    features = features or CLUSTERING_FEATURES
    X = df[features].copy()

    skew_before = X.skew().to_dict()

    pt = PowerTransformer(method="yeo-johnson")
    skewed_present = [f for f in SKEWED_FEATURES if f in features]
    X_transformed = X.copy()
    if skewed_present:
        X_transformed[skewed_present] = pt.fit_transform(X[skewed_present])

    skew_after = X_transformed.skew().to_dict()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_transformed), columns=features, index=df.index
    )

    meta = {
        "features": features,
        "skew_before": skew_before,
        "skew_after": skew_after,
        "power_transformer": pt,
        "scaler": scaler,
    }
    return X_scaled, meta


def elbow_analysis(X, k_range=range(2, 11)):
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(X)
        inertias[k] = km.inertia_
    return inertias


def silhouette_analysis(X, k_range=range(2, 11), sample_size=3000):
    scores = {}
    rng = np.random.default_rng(RANDOM_SEED)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample_base = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_sample_base = X

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels_full = km.fit_predict(X)
        if len(X) > sample_size:
            labels_sample = labels_full[idx]
            score = silhouette_score(X_sample_base, labels_sample)
        else:
            score = silhouette_score(X, labels_full)
        scores[k] = score
    return scores


def stability_analysis(X, k, n_bootstrap=10, sample_frac=0.8):
    """Bootstrap stability check: refit K-Means on random subsamples and
    measure the Adjusted Rand Index between the reference labelling (fit on
    full data) and each bootstrap labelling restricted to the overlapping
    customers. High mean ARI (closer to 1) indicates the chosen k produces a
    reproducible partition rather than an artefact of one particular sample.
    """
    from sklearn.metrics import adjusted_rand_score

    km_full = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    ref_labels = km_full.fit_predict(X)

    n = len(X)
    ari_scores = []
    rng = np.random.default_rng(RANDOM_SEED)
    X_arr = X.values if hasattr(X, "values") else X

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=int(n * sample_frac), replace=False)
        km_boot = KMeans(n_clusters=k, random_state=RANDOM_SEED + i, n_init=10)
        boot_labels = km_boot.fit_predict(X_arr[idx])
        ari = adjusted_rand_score(ref_labels[idx], boot_labels)
        ari_scores.append(ari)

    return {"mean_ari": float(np.mean(ari_scores)), "std_ari": float(np.std(ari_scores)), "scores": ari_scores}


def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def fit_hierarchical(X, k, linkage="ward"):
    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = agg.fit_predict(X)
    return agg, labels


def compare_algorithms(X, k):
    from sklearn.metrics import adjusted_rand_score

    km, km_labels = fit_kmeans(X, k)
    agg, agg_labels = fit_hierarchical(X, k)

    sample_n = min(3000, len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=sample_n, replace=False)
    X_sample = X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    results = {
        "kmeans": {
            "silhouette": silhouette_score(X_sample, km_labels[idx]),
            "calinski_harabasz": calinski_harabasz_score(X, km_labels),
            "davies_bouldin": davies_bouldin_score(X, km_labels),
        },
        "hierarchical": {
            "silhouette": silhouette_score(X_sample, agg_labels[idx]),
            "calinski_harabasz": calinski_harabasz_score(X, agg_labels),
            "davies_bouldin": davies_bouldin_score(X, agg_labels),
        },
        "agreement_ari": adjusted_rand_score(km_labels, agg_labels),
    }
    return results, km_labels, agg_labels


def run_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    components = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T, index=X.columns if hasattr(X, "columns") else None,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return pca, components, explained, loadings


if __name__ == "__main__":
    import preprocessing as pp
    import feature_engineering as fe

    customers, transactions = pp.load_raw("../data/raw/customers.csv", "../data/raw/transactions.csv")
    customers_clean, transactions_clean, log = pp.validate_and_clean(customers, transactions)
    features = fe.build_customer_features(customers_clean, transactions_clean, snapshot_date="2025-01-31")

    X, meta = prepare_clustering_matrix(features)
    print("Scaled matrix shape:", X.shape)

    inertias = elbow_analysis(X, range(2, 9))
    print("Inertias:", inertias)

    sils = silhouette_analysis(X, range(2, 9))
    print("Silhouettes:", sils)

    best_k = max(sils, key=sils.get)
    print("Best k by silhouette:", best_k)

    stability = stability_analysis(X, best_k)
    print("Stability:", stability)

    comparison, km_labels, agg_labels = compare_algorithms(X, best_k)
    print("Comparison:", comparison)
