"""
preprocessing.py
================
Data validation and cleaning for the Customer Value Segmentation project.

Handles:
    * duplicate detection & removal
    * missing-value analysis and treatment
    * categorical inconsistency normalisation
    * invalid / negative value correction
    * outlier detection (IQR + z-score) with documented treatment decisions

Every cleaning decision is logged into a dict returned alongside the cleaned
data, so the pipeline can report exactly what was changed and why (used in
reports/quality_control_report.md).
"""

import numpy as np
import pandas as pd


def load_raw(customers_path, transactions_path):
    customers = pd.read_csv(customers_path, parse_dates=["signup_date"])
    transactions = pd.read_csv(transactions_path, parse_dates=["order_date"])
    return customers, transactions


def validate_and_clean(customers: pd.DataFrame, transactions: pd.DataFrame):
    log = {}

    # ---------------------------------------------------------------
    # 1. Duplicate detection
    # ---------------------------------------------------------------
    n_dup_customers = customers.duplicated().sum()
    customers = customers.drop_duplicates().copy()

    n_dup_tx = transactions.duplicated().sum()
    transactions = transactions.drop_duplicates().copy()

    log["duplicate_customer_rows_removed"] = int(n_dup_customers)
    log["duplicate_transaction_rows_removed"] = int(n_dup_tx)

    # Duplicate customer_ids that are NOT full-row duplicates (edge case check)
    dup_ids = customers["customer_id"][customers["customer_id"].duplicated()]
    if len(dup_ids) > 0:
        # keep first occurrence
        customers = customers.drop_duplicates(subset="customer_id", keep="first").copy()
    log["duplicate_customer_ids_collapsed"] = int(len(dup_ids))

    # ---------------------------------------------------------------
    # 2. Categorical normalisation (case inconsistency)
    # ---------------------------------------------------------------
    customers["region"] = customers["region"].astype(str).str.strip().str.title()
    customers.loc[customers["region"] == "Nan", "region"] = np.nan

    transactions["channel"] = transactions["channel"].astype(str).str.strip().str.title()

    # ---------------------------------------------------------------
    # 3. Missing-value analysis
    # ---------------------------------------------------------------
    missing_summary_customers = customers.isna().mean().round(4) * 100
    missing_summary_tx = transactions.isna().mean().round(4) * 100
    log["missing_pct_customers"] = missing_summary_customers[missing_summary_customers > 0].to_dict()
    log["missing_pct_transactions"] = missing_summary_tx[missing_summary_tx > 0].to_dict()

    # Treatment decisions:
    #   - region: keep as "Unknown" category (categorical, low % missing, MCAR-like -> safe to flag)
    #   - age: median imputation, flagged with an indicator column (numeric, ~4% missing)
    #   - avg_satisfaction_score: median imputation + indicator (customers who never took a survey
    #     are not necessarily different in value -> flag rather than assume MCAR blindly)
    #   - transactions.order_value missing: DROP the row. We cannot impute a monetary transaction
    #     value without materially fabricating revenue; missing rows are <1% of data (safe to drop).
    customers["region"] = customers["region"].fillna("Unknown")

    customers["age_missing_flag"] = customers["age"].isna().astype(int)
    customers["age"] = customers["age"].fillna(customers["age"].median())

    customers["satisfaction_missing_flag"] = customers["avg_satisfaction_score"].isna().astype(int)
    customers["avg_satisfaction_score"] = customers["avg_satisfaction_score"].fillna(
        customers["avg_satisfaction_score"].median()
    )

    n_before = len(transactions)
    transactions = transactions.dropna(subset=["order_value"]).copy()
    log["transaction_rows_dropped_missing_value"] = int(n_before - len(transactions))

    # ---------------------------------------------------------------
    # 4. Invalid values
    # ---------------------------------------------------------------
    n_negative = (transactions["order_value"] < 0).sum()
    # Negative order_value with is_return == True would be a legitimate refund;
    # here they were injected as data-entry errors alongside non-return flags,
    # so we treat all negative order_values as logging errors and correct the sign.
    transactions.loc[transactions["order_value"] < 0, "order_value"] = transactions.loc[
        transactions["order_value"] < 0, "order_value"
    ].abs()
    log["negative_order_values_corrected"] = int(n_negative)

    n_zero_or_neg_age = (customers["age"] <= 0).sum()
    customers = customers[customers["age"] > 0].copy()
    log["invalid_age_rows_removed"] = int(n_zero_or_neg_age)

    # ---------------------------------------------------------------
    # 5. Outlier analysis (IQR method) on order_value
    # ---------------------------------------------------------------
    q1, q3 = transactions["order_value"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 3 * iqr  # 3x IQR (extreme-outlier convention) to avoid over-trimming genuine big baskets
    lower_fence = max(0, q1 - 3 * iqr)

    n_extreme_outliers = (transactions["order_value"] > upper_fence).sum()
    log["order_value_iqr_bounds"] = {"lower": round(float(lower_fence), 2), "upper": round(float(upper_fence), 2)}
    log["extreme_order_value_outliers_flagged"] = int(n_extreme_outliers)

    # Decision: CAP (winsorise) rather than drop -> these are plausible large
    # baskets/whales for a retail business and dropping them would bias
    # Monetary features downward for genuinely high-value customers. Capping
    # preserves their "high value" signal while limiting undue leverage on
    # distance-based clustering.
    transactions["order_value_capped"] = np.where(
        transactions["order_value"] > upper_fence, upper_fence, transactions["order_value"]
    )

    # Remove any customer_id referenced in transactions but absent from the
    # (cleaned) customer master, and vice versa flag customers with 0 orders
    # (still valid — they are simply inactive / never purchased).
    valid_customers = set(customers["customer_id"])
    n_orphan_tx = (~transactions["customer_id"].isin(valid_customers)).sum()
    transactions = transactions[transactions["customer_id"].isin(valid_customers)].copy()
    log["orphan_transactions_removed"] = int(n_orphan_tx)

    log["final_customer_count"] = int(len(customers))
    log["final_transaction_count"] = int(len(transactions))

    return customers, transactions, log


if __name__ == "__main__":
    customers, transactions = load_raw("../data/raw/customers.csv", "../data/raw/transactions.csv")
    customers_clean, transactions_clean, log = validate_and_clean(customers, transactions)
    print(log)
