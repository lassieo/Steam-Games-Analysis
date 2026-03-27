# Issues And Challenges

### 1. Dataset limitations reduce completeness
The current dataset does not directly capture revenue, profit, development budget, marketing spend, or long-term retention. That means success is being measured from platform-visible engagement rather than full commercial performance.

### 2. Some provided review fields are inconsistent
Earlier analysis already showed that `num_reviews_total` and `pct_pos_total` are inconsistent with the raw `positive` and `negative` counts in many rows. Because of that, the project had to rebuild review totals from the raw counts instead of trusting all precomputed fields.

### 3. Price analysis is affected by extreme outliers
The price distribution includes very unusual titles with very high listed prices such as `$199.99`, `$500.00`, and `$999.98`. These outliers can distort summary statistics and make the highest-price group harder to interpret.

### 4. Free games are difficult to compare directly with paid games
Free games may follow different player acquisition patterns, monetization models, and review behaviors. Comparing them directly to paid games under one success metric can flatten meaningful differences.

### 5. Cross-sectional data limits the conclusions
This dataset acts like a snapshot. It does not show how a game's price, reviews, or success status changed over time. Because of that, the analysis is much better at finding patterns than proving timing or causality.

## Summary
The main goal for the final report is to present clear evidence-based patterns while briefly acknowledging dataset constraints. The current success rule can remain fixed, and the focus can stay on interpreting the existing results rather than redesigning the target.
