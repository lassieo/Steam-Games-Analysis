# Steam Success Metric Report

## Scope
This report uses the latest cleaned file: `games_march2025_cleaned.csv` (89,618 games).

## Candidate Variables For Success
Recommended candidate variables:

1. `positive` - raw count of positive user reviews.
2. `negative` - raw count of negative user reviews.
3. `review_total_calc = positive + negative` - reconstructed review volume.
4. `positive_ratio_calc = positive / (positive + negative)` - reconstructed sentiment quality.
5. `recommendations` - optional engagement cross-check, useful for interpretation but not required in the final label.

Interpretation:

- "Good" is best captured by a high positive review ratio.
- "Popular" is best captured by a minimum amount of review volume.
- `pct_pos_total` and `num_reviews_total` were not used for the final formula because they are inconsistent with `positive` and `negative` in many rows.

Data quality note:

- Rows where `num_reviews_total != positive + negative`: 83,401
- Rows where `num_reviews_total > 0` but `positive + negative = 0`: 9,151

##  Candidate Success Definitions
### Option 1 (Simple)
Plain English: A game is successful if at least 90% of its reviews are positive.

Pseudo code:
`success = 1 if positive_ratio_calc >= 0.90 else 0`

### (Balanced)
Plain English: A game is successful if at least 85% of its reviews are positive and it has at least 50 total reviews.

Pseudo code:
`success = 1 if positive_ratio_calc >= 0.85 and review_total_calc >= 50 else 0`

### Option 3 (Relative)
Plain English: A game is successful if it is in the top 25% of reviewed games for both sentiment and review volume.

Pseudo code:
`success = 1 if positive_ratio_calc >= 0.9412 and review_total_calc >= 127 else 0`

## Day 3 - Comparison Of Options
| Option | Successful Games | % of Dataset | Included Review Count | Interpretation |
|---|---:|---:|---|---|
| Option 1 (Simple) | 24,405 | 27.23% | Min 1, median 13 | Too lenient. It includes many games with only 1-3 reviews, so it over-labels tiny games as successful. |
| Option 2 (Balanced) | 11,068 | 12.35% | Min 50, median 328 | Best balance. It removes tiny low-evidence games while keeping a usable number of positives. |
| Option 3 (Relative) | 2,401 | 2.68% | Min 127, median 853 | Too strict for a binary target. It keeps only an elite slice of games and risks class imbalance. |

## Day 4 - Final Definition
Selected definition: **Option 2 (Balanced)**

Final formula:

`success = 1 if positive_ratio_calc >= 0.85 and review_total_calc >= 50 else 0`

Why this works:

- It captures quality through strong sentiment.
- It captures popularity through minimum review evidence.
- It avoids the biggest failure of Option 1, which is labeling one-review games as successful.
- It avoids the biggest failure of Option 3, which is making success so exclusive that the positive class becomes too small.

Why the others were rejected:

- Option 1 was rejected because it is too lenient and includes many low-evidence games.
- Option 3 was rejected because it is too strict and leaves a relatively small positive class.

## Implemented Success Variable
The output file includes:

- `review_total_calc`
- `positive_ratio_calc`
- `has_reviews_calc`
- `success`

Class balance in the final output:

- Successful games: 11,068
- Unsuccessful games: 78,550
- Positive class share: 12.35%

## Documentation Write-Up
### Final definition
Success is defined as having at least 85% positive reviews and at least 50 total reviews, where total reviews are calculated as `positive + negative`.

### Reasoning
This definition balances product quality and market traction. A high positive ratio alone is not enough, because very small games can reach 100% positive with only a handful of reviews. Adding a minimum review threshold makes the label more credible while preserving enough successful games for downstream analysis or modeling.

### Limitations
1. Review volume is only one form of popularity; some games may be commercially successful with fewer Steam reviews.
2. The 50-review threshold is still somewhat judgment-based, even though it improves on the simple ratio-only rule.
3. The source file contains inconsistent precomputed review fields, so the analysis relies on `positive` and `negative` instead.
4. This is a Steam-platform success proxy, not a complete measure of revenue or long-term impact.
