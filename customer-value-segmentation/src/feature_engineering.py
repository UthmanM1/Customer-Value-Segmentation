"""
feature_engineering.py
=======================
Aggregates cleaned transaction-level data to customer level and constructs
RFM (Recency, Frequency, Monetary) variables plus additional behavioural
features used for segmentation.

All engineered features are documented in reports/data_dictionary.md.
"""

import numpy as np
import pandas as pd


def build_customer_features(customers: pd.DataFrame, transactions: pd.DataFrame, snapshot_date):
    """Aggregate transactions to one row per customer and merge with the
    customer master table.
    """
    snapshot_date = pd.Timestamp(snapshot_date)

    tx = transactions.copy()
    tx["is_return"] = tx["is_return"].astype(bool)

    # --- Core RFM ---------------------------------------------------
    grp = tx.groupby("customer_id")

    recency = (snapshot_date - grp["order_date"].max()).dt.days.rename("recency_days")
    frequency = grp.size().rename("frequency")
    monetary_total = grp["order_value_capped"].sum().rename("monetary_total")
    monetary_avg = grp["order_value_capped"].mean().rename("avg_order_value")
    monetary_median = grp["order_value_capped"].median().rename("median_order_value")
    order_value_std = grp["order_value_capped"].std().fillna(0).rename("order_value_std")

    # --- Additional behavioural features ------------------------------
    first_order = grp["order_date"].min().rename("first_order_date")
    last_order = grp["order_date"].max().rename("last_order_date")

    active_span_days = (last_order - first_order).dt.days.clip(lower=0).rename("active_span_days")

    n_categories = grp["product_category"].nunique().rename("distinct_categories_purchased")

    return_rate = grp["is_return"].mean().rename("return_rate")

    channel_counts = tx.groupby(["customer_id", "channel"]).size().unstack(fill_value=0)
    channel_counts.columns = [f"orders_via_{c.lower().replace(' ', '_')}" for c in channel_counts.columns]

    # favourite category (mode) -> useful for qualitative profiling, not for clustering directly
    fav_category = tx.groupby("customer_id")["product_category"].agg(lambda s: s.value_counts().idxmax())
    fav_category.name = "favourite_category"

    features = pd.concat([
        recency, frequency, monetary_total, monetary_avg, monetary_median,
        order_value_std, active_span_days, n_categories, return_rate, fav_category
    ], axis=1)

    features = features.join(channel_counts, how="left")
    features = features.reset_index()

    # --- Merge with customer master (includes customers with ZERO orders) ---
    full = customers.merge(features, on="customer_id", how="left")

    # Customers with no transactions in the window: fill purchase-derived fields sensibly
    never_purchased = full["frequency"].isna()
    full["frequency"] = full["frequency"].fillna(0)
    full["monetary_total"] = full["monetary_total"].fillna(0)
    full["avg_order_value"] = full["avg_order_value"].fillna(0)
    full["median_order_value"] = full["median_order_value"].fillna(0)
    full["order_value_std"] = full["order_value_std"].fillna(0)
    full["distinct_categories_purchased"] = full["distinct_categories_purchased"].fillna(0)
    full["return_rate"] = full["return_rate"].fillna(0)
    full["active_span_days"] = full["active_span_days"].fillna(0)
    # recency for non-purchasers: distance from signup to snapshot (they have never engaged)
    full.loc[never_purchased, "recency_days"] = (
        snapshot_date - full.loc[never_purchased, "signup_date"]
    ).dt.days
    for c in [col for col in full.columns if col.startswith("orders_via_")]:
        full[c] = full[c].fillna(0)
    full["favourite_category"] = full["favourite_category"].fillna("None")
    full["never_purchased_flag"] = never_purchased.astype(int)

    # --- Tenure & derived ratios --------------------------------------
    full["tenure_days"] = (snapshot_date - full["signup_date"]).dt.days
    full["tenure_months"] = (full["tenure_days"] / 30.44).round(1)

    # Purchase frequency normalised by tenure (orders per active month) —
    # avoids penalising newer customers purely for having less history.
    full["orders_per_tenure_month"] = full["frequency"] / full["tenure_months"].clip(lower=1)

    # Monetary value normalised similarly
    full["spend_per_tenure_month"] = full["monetary_total"] / full["tenure_months"].clip(lower=1)

    # Engagement composite: combines opt-in/loyalty binary signals
    full["engagement_score"] = (
        full["newsletter_opt_in"].astype(int) + full["loyalty_member"].astype(int)
    )

    # Support burden ratio (tickets per order, guards against div/0)
    full["support_tickets_per_order"] = full["support_tickets_12m"] / full["frequency"].clip(lower=1)

    return full


DATA_DICTIONARY = {
    "customer_id": "Unique customer identifier",
    "signup_date": "Date the customer first registered",
    "region": "UK region (normalised categorical)",
    "acquisition_channel": "Marketing channel that acquired the customer",
    "age": "Customer age in years (median-imputed where missing)",
    "age_missing_flag": "1 if age was imputed",
    "newsletter_opt_in": "Whether customer opted into marketing emails",
    "loyalty_member": "Whether customer is enrolled in the loyalty programme",
    "support_tickets_12m": "Number of customer support tickets raised in the last 12 months",
    "avg_satisfaction_score": "Average CSAT score (1-5), median-imputed where missing",
    "satisfaction_missing_flag": "1 if satisfaction score was imputed",
    "recency_days": "Days since the customer's most recent order (RFM: Recency)",
    "frequency": "Total number of orders placed in the analysis window (RFM: Frequency)",
    "monetary_total": "Total spend, outlier-capped (RFM: Monetary)",
    "avg_order_value": "Mean order value per customer",
    "median_order_value": "Median order value per customer (robust to outliers)",
    "order_value_std": "Standard deviation of order values (spend consistency)",
    "active_span_days": "Days between a customer's first and last order",
    "distinct_categories_purchased": "Count of unique product categories purchased",
    "return_rate": "Proportion of orders that were returns",
    "favourite_category": "Most frequently purchased product category",
    "never_purchased_flag": "1 if the customer has never placed an order",
    "tenure_days": "Days since signup",
    "tenure_months": "Months since signup",
    "orders_per_tenure_month": "Order frequency normalised by tenure",
    "spend_per_tenure_month": "Spend normalised by tenure",
    "engagement_score": "Simple composite of newsletter opt-in + loyalty membership (0-2)",
    "support_tickets_per_order": "Support tickets raised per order placed (service burden)",
}


if __name__ == "__main__":
    import preprocessing as pp
    customers, transactions = pp.load_raw("../data/raw/customers.csv", "../data/raw/transactions.csv")
    customers_clean, transactions_clean, log = pp.validate_and_clean(customers, transactions)
    features = build_customer_features(customers_clean, transactions_clean, snapshot_date="2025-01-31")
    print(features.shape)
    print(features.head())
