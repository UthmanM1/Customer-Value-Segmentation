# Data Dictionary

All fields used in the Customer Value Segmentation analysis, generated from `data/processed/customer_features.csv`. Report generated 2026-09-06.

| Field | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `signup_date` | Date the customer first registered |
| `region` | UK region (normalised categorical) |
| `acquisition_channel` | Marketing channel that acquired the customer |
| `age` | Customer age in years (median-imputed where missing) |
| `age_missing_flag` | 1 if age was imputed |
| `newsletter_opt_in` | Whether customer opted into marketing emails |
| `loyalty_member` | Whether customer is enrolled in the loyalty programme |
| `support_tickets_12m` | Number of customer support tickets raised in the last 12 months |
| `avg_satisfaction_score` | Average CSAT score (1-5), median-imputed where missing |
| `satisfaction_missing_flag` | 1 if satisfaction score was imputed |
| `recency_days` | Days since the customer's most recent order (RFM: Recency) |
| `frequency` | Total number of orders placed in the analysis window (RFM: Frequency) |
| `monetary_total` | Total spend, outlier-capped (RFM: Monetary) |
| `avg_order_value` | Mean order value per customer |
| `median_order_value` | Median order value per customer (robust to outliers) |
| `order_value_std` | Standard deviation of order values (spend consistency) |
| `active_span_days` | Days between a customer's first and last order |
| `distinct_categories_purchased` | Count of unique product categories purchased |
| `return_rate` | Proportion of orders that were returns |
| `favourite_category` | Most frequently purchased product category |
| `never_purchased_flag` | 1 if the customer has never placed an order |
| `tenure_days` | Days since signup |
| `tenure_months` | Months since signup |
| `orders_per_tenure_month` | Order frequency normalised by tenure |
| `spend_per_tenure_month` | Spend normalised by tenure |
| `engagement_score` | Simple composite of newsletter opt-in + loyalty membership (0-2) |
| `support_tickets_per_order` | Support tickets raised per order placed (service burden) |

## Notes
- `segment` / `segment_name`: assigned by the final K-Means model (k=4) described in `segmentation_report.md`.
- `pca_1` / `pca_2`: first two principal components used only for visualisation (see the PCA section of the segmentation report for what they do and do not represent).