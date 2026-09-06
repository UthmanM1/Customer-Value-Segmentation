"""
data_generation.py
===================
Generates a realistic SYNTHETIC customer + transaction dataset for a UK
subscription/retail business, for the Customer Value Segmentation portfolio
project.

Design goals (per project brief):
    * 10,000+ customers, 12+ months of transaction history
    * Realistic data-quality issues: missing values, duplicate rows,
      inconsistent formatting, a handful of extreme outliers
    * Overlapping / fuzzy customer behaviour groups rather than obviously
      separated clusters -> genuine analytical work is required downstream
    * Fully reproducible via a fixed random seed

This dataset is entirely synthetic. It does not represent any real company,
customer, or transaction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RANDOM_SEED = 42
N_CUSTOMERS = 10500
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 1, 31)  # ~13 months of trading history
SNAPSHOT_DATE = END_DATE  # "today" for recency calculations downstream

PRODUCT_CATEGORIES = [
    "Homeware", "Fashion", "Electronics", "Beauty", "Groceries",
    "Books & Media", "Sports & Outdoors", "Toys & Kids", "Garden", "Pet Supplies"
]

UK_REGIONS = [
    "London", "South East", "North West", "West Midlands", "Scotland",
    "Yorkshire", "East of England", "South West", "Wales", "North East"
]

ACQUISITION_CHANNELS = ["Paid Search", "Organic", "Social", "Email", "Referral", "Affiliate"]


def _rng():
    return np.random.default_rng(RANDOM_SEED)


def generate_customers(n=N_CUSTOMERS, rng=None):
    """Generate the customer master table.

    Customers are drawn from a small number of LATENT (unlabelled) behavioural
    archetypes with substantial noise and overlap, so that no single raw
    feature cleanly separates them. This mimics real-world customer bases
    where true segments must be discovered rather than read off a label.
    """
    if rng is None:
        rng = _rng()

    customer_id = np.arange(100000, 100000 + n)

    # Latent archetype mixture (NOT exposed as a feature; purely a generative
    # device to create realistic, overlapping behavioural structure).
    archetype_probs = [0.14, 0.22, 0.30, 0.19, 0.15]
    archetype = rng.choice(5, size=n, p=archetype_probs)
    # 0: Champions (high value, frequent, long tenure)
    # 1: Steady mid-value
    # 2: Low-engagement / bargain browsers
    # 3: New / recently acquired
    # 4: High value but fading (previously active, now quiet)

    # Tenure (days since signup), correlated with archetype but with heavy overlap
    tenure_base = {
        0: 620, 1: 420, 2: 300, 3: 60, 4: 560
    }
    tenure_days = np.array([
        max(7, rng.normal(tenure_base[a], 180)) for a in archetype
    ])
    tenure_days = np.clip(tenure_days, 1, 1500)
    signup_date = pd.Series([END_DATE - timedelta(days=int(t)) for t in tenure_days])

    region = rng.choice(UK_REGIONS, size=n)
    channel = rng.choice(ACQUISITION_CHANNELS, size=n, p=[0.28, 0.22, 0.18, 0.14, 0.1, 0.08])

    age = np.clip(rng.normal(41, 13, size=n), 18, 85).round().astype(int)

    newsletter_opt_in = rng.random(n) < 0.55
    loyalty_member = rng.random(n) < 0.38

    # Support interactions (weakly related to archetype: fading/at-risk types
    # tend to raise slightly more tickets before churn signals appear)
    support_lambda = np.select(
        [archetype == 4, archetype == 0, archetype == 2],
        [1.6, 0.7, 0.9],
        default=1.0
    )
    support_tickets = rng.poisson(support_lambda)

    avg_satisfaction = np.clip(
        rng.normal(np.select([archetype == 4], [3.2], default=4.0), 0.9, size=n), 1, 5
    ).round(1)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "signup_date": signup_date,
        "region": region,
        "acquisition_channel": channel,
        "age": age,
        "newsletter_opt_in": newsletter_opt_in,
        "loyalty_member": loyalty_member,
        "support_tickets_12m": support_tickets,
        "avg_satisfaction_score": avg_satisfaction,
        "_archetype": archetype,  # kept privately for QA only; dropped before saving raw file downstream if desired
    })
    return df


def generate_transactions(customers, rng=None):
    """Generate a transaction-level log for each customer.

    Purchase frequency, order value, category mix and recency of last
    purchase are all driven by the latent archetype but with large individual
    variance, seasonal effects, and cross-cutting noise so that behavioural
    groups overlap substantially.
    """
    if rng is None:
        rng = _rng()

    records = []

    # Archetype -> (mean orders per active month, order value mean, order value sd,
    #               probability the customer has "gone quiet" in the last 90 days)
    archetype_params = {
        0: dict(rate=2.2, val_mu=68, val_sd=28, quiet_p=0.05, returns_p=0.06),
        1: dict(rate=1.1, val_mu=42, val_sd=18, quiet_p=0.15, returns_p=0.08),
        2: dict(rate=0.45, val_mu=24, val_sd=12, quiet_p=0.35, returns_p=0.10),
        3: dict(rate=0.9, val_mu=38, val_sd=20, quiet_p=0.20, returns_p=0.07),
        4: dict(rate=1.6, val_mu=60, val_sd=25, quiet_p=0.55, returns_p=0.09),  # was active, now fading
    }

    seasonal_boost_month = {11: 1.35, 12: 1.55, 1: 0.85}  # Nov/Dec peak, Jan dip (UK retail pattern)

    for _, cust in customers.iterrows():
        params = archetype_params[cust["_archetype"]]
        signup = cust["signup_date"]
        active_start = max(signup, START_DATE)
        months_active = max(1, (END_DATE.year - active_start.year) * 12 + (END_DATE.month - active_start.month) + 1)

        # "Going quiet" effect: a fraction of customers per archetype stop
        # purchasing partway through the window (drives Recency variance)
        goes_quiet = rng.random() < params["quiet_p"]
        quiet_from_month = None
        if goes_quiet:
            quiet_from_month = rng.integers(1, max(2, months_active))

        month_cursor = active_start.replace(day=1)
        month_idx = 0
        while month_cursor <= END_DATE:
            month_idx += 1
            if quiet_from_month is not None and month_idx >= quiet_from_month:
                # sharply reduced probability of any purchase after going quiet
                month_rate = params["rate"] * 0.05
            else:
                month_rate = params["rate"]

            season_mult = seasonal_boost_month.get(month_cursor.month, 1.0)
            n_orders_this_month = rng.poisson(max(0.01, month_rate * season_mult))

            for _ in range(n_orders_this_month):
                day_offset = rng.integers(0, 28)
                order_date = month_cursor + timedelta(days=int(day_offset))
                if order_date > END_DATE:
                    continue
                order_value = max(3.0, rng.normal(params["val_mu"], params["val_sd"]))
                # occasional high-value basket (realistic long tail)
                if rng.random() < 0.02:
                    order_value *= rng.uniform(3, 8)

                category = rng.choice(PRODUCT_CATEGORIES)
                is_return = rng.random() < params["returns_p"]

                records.append({
                    "customer_id": cust["customer_id"],
                    "order_date": order_date,
                    "order_value": round(order_value, 2),
                    "product_category": category,
                    "is_return": is_return,
                    "channel": rng.choice(["Web", "Mobile App", "Store"], p=[0.55, 0.35, 0.10]),
                })

            # move to next month
            if month_cursor.month == 12:
                month_cursor = month_cursor.replace(year=month_cursor.year + 1, month=1)
            else:
                month_cursor = month_cursor.replace(month=month_cursor.month + 1)

    tx = pd.DataFrame.from_records(records)
    return tx


def inject_data_quality_issues(customers, transactions, rng=None):
    """Introduce realistic messiness: missing values, duplicates,
    inconsistent categorical formatting, and a few extreme outliers.
    This is deliberate and mirrors real operational data.
    """
    if rng is None:
        rng = _rng()

    customers = customers.copy()
    transactions = transactions.copy()

    # --- Missing values -------------------------------------------------
    # Age missing for ~4% of customers
    miss_age_idx = customers.sample(frac=0.04, random_state=RANDOM_SEED).index
    customers.loc[miss_age_idx, "age"] = np.nan

    # Satisfaction score missing for ~12% (customers who never responded to a survey)
    miss_sat_idx = customers.sample(frac=0.12, random_state=RANDOM_SEED + 1).index
    customers.loc[miss_sat_idx, "avg_satisfaction_score"] = np.nan

    # Region missing for ~2%
    miss_region_idx = customers.sample(frac=0.02, random_state=RANDOM_SEED + 2).index
    customers.loc[miss_region_idx, "region"] = np.nan

    # A few order_values missing (failed payment logging) ~0.5%
    miss_val_idx = transactions.sample(frac=0.005, random_state=RANDOM_SEED + 3).index
    transactions.loc[miss_val_idx, "order_value"] = np.nan

    # --- Inconsistent categorical formatting -----------------------------
    region_case_noise = customers.sample(frac=0.05, random_state=RANDOM_SEED + 4).index
    customers.loc[region_case_noise, "region"] = customers.loc[region_case_noise, "region"].astype(str).str.upper()

    channel_noise_idx = transactions.sample(frac=0.03, random_state=RANDOM_SEED + 5).index
    transactions.loc[channel_noise_idx, "channel"] = transactions.loc[channel_noise_idx, "channel"].astype(str).str.lower()

    # --- Duplicate rows (system double-logging) --------------------------
    dup_rows = transactions.sample(frac=0.008, random_state=RANDOM_SEED + 6)
    transactions = pd.concat([transactions, dup_rows], ignore_index=True)

    dup_customers = customers.sample(frac=0.003, random_state=RANDOM_SEED + 7)
    customers = pd.concat([customers, dup_customers], ignore_index=True)

    # --- Extreme outliers (data-entry errors / genuine whales) -----------
    outlier_idx = transactions.sample(n=15, random_state=RANDOM_SEED + 8).index
    transactions.loc[outlier_idx, "order_value"] = transactions.loc[outlier_idx, "order_value"] * rng.uniform(15, 40, size=15)

    # A handful of negative order values (refund logging error, should be caught in cleaning)
    neg_idx = transactions.sample(n=8, random_state=RANDOM_SEED + 9).index
    transactions.loc[neg_idx, "order_value"] = -abs(transactions.loc[neg_idx, "order_value"])

    return customers, transactions


def main(output_dir="../data/raw"):
    rng = _rng()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating customer master table...")
    customers = generate_customers(rng=rng)

    print("Generating transaction log (this may take a moment)...")
    transactions = generate_transactions(customers, rng=rng)

    print("Injecting realistic data-quality issues...")
    customers_dirty, transactions_dirty = inject_data_quality_issues(customers, transactions, rng=rng)

    # Drop the latent archetype label before saving raw data — this must be
    # rediscovered by the analysis, not read off a hidden column.
    customers_public = customers_dirty.drop(columns=["_archetype"])

    customers_public.to_csv(out / "customers.csv", index=False)
    transactions_dirty.to_csv(out / "transactions.csv", index=False)

    print(f"Saved {len(customers_public):,} customer rows -> {out/'customers.csv'}")
    print(f"Saved {len(transactions_dirty):,} transaction rows -> {out/'transactions.csv'}")

    return customers_public, transactions_dirty


if __name__ == "__main__":
    main()
