# Statistical Validation Report

*Report generated 2026-09-06 from live pipeline output.*

## Method

For each candidate metric, a **Kruskal-Wallis H-test** was used to test whether the metric's distribution differs across the four segments. Kruskal-Wallis (a non-parametric alternative to one-way ANOVA) was chosen because RFM variables are right-skewed and clearly non-normal (see `segmentation_report.md`), violating the normality assumption behind ANOVA. Effect size is reported as **epsilon-squared (ε²)**, the non-parametric analogue of eta-squared, interpreted as: <0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, ≥0.14 large. Because 13 metrics were tested simultaneously, p-values were corrected for multiple comparisons using the **Benjamini-Hochberg false discovery rate (FDR)** procedure (α = 0.05) rather than reporting raw p-values, which would inflate the false-positive rate.

## Results

| metric                        |   h_statistic |   p_value |   p_value_fdr_corrected | significant_after_correction   |   effect_size_epsilon_sq | effect_size_label   |
|:------------------------------|--------------:|----------:|------------------------:|:-------------------------------|-------------------------:|:--------------------|
| monetary_total                |        8622.4 |  0        |                0        | True                           |                    0.821 | large               |
| frequency                     |        8550.2 |  0        |                0        | True                           |                    0.814 | large               |
| distinct_categories_purchased |        8336.2 |  0        |                0        | True                           |                    0.794 | large               |
| spend_per_tenure_month        |        8075.9 |  0        |                0        | True                           |                    0.769 | large               |
| orders_per_tenure_month       |        7536.5 |  0        |                0        | True                           |                    0.718 | large               |
| avg_order_value               |        6360.7 |  0        |                0        | True                           |                    0.606 | large               |
| median_order_value            |        6277.9 |  0        |                0        | True                           |                    0.598 | large               |
| tenure_months                 |        5936   |  0        |                0        | True                           |                    0.565 | large               |
| recency_days                  |        3268   |  0        |                0        | True                           |                    0.311 | large               |
| return_rate                   |        1821   |  0        |                0        | True                           |                    0.173 | large               |
| avg_satisfaction_score        |          71.6 |  1.93e-15 |                2.28e-15 | True                           |                    0.007 | negligible          |
| engagement_score              |           6.2 |  0.1026   |                0.1112   | False                          |                    0     | negligible          |
| support_tickets_12m           |           4.2 |  0.2374   |                0.2374   | False                          |                    0     | negligible          |

**11 of 13** tested metrics remained statistically significant after FDR correction.

## Practical Significance

The following metrics show a **large** effect size, meaning the segments differ not just statistically but substantially in practice: `monetary_total`, `frequency`, `distinct_categories_purchased`, `spend_per_tenure_month`, `orders_per_tenure_month`, `avg_order_value`, `median_order_value`, `tenure_months`, `recency_days`, `return_rate`.

The following metrics show a **negligible** effect size despite the clustering: `avg_satisfaction_score`, `engagement_score`, `support_tickets_12m`. This is expected and important to report honestly: these variables were largely unrelated to the RFM/behavioural feature set the clustering was built on, so segments should not be assumed to differ meaningfully on them, and business strategies should not rely on these variables as segment-distinguishing traits.

## Caveat

With a sample size in the thousands per segment, even practically trivial differences can become statistically significant (p < 0.05). This is exactly why effect size, not p-value alone, is used here to judge whether a difference matters for business decision-making.