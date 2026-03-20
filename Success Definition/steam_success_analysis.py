from pathlib import Path

import pandas as pd


BASE_DIR = Path("/Users/laasyavenugopal/Desktop/Yelp JSON/yelp_dataset")
SOURCE_FILE = Path("/Users/laasyavenugopal/Desktop/Steam Datasets/games_march2025_cleaned.csv")
OUTPUT_CSV = BASE_DIR / "games_march2025_with_success.csv"
OUTPUT_MD = BASE_DIR / "steam_success_report.md"


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_num(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{int(value):,}"


def main() -> None:
    df = pd.read_csv(SOURCE_FILE, low_memory=False)

    # Use raw positive/negative counts because the provided total review fields
    # are inconsistent for many games in this dataset.
    df["review_total_calc"] = df["positive"].fillna(0) + df["negative"].fillna(0)
    df["positive_ratio_calc"] = df["positive"] / df["review_total_calc"]
    df.loc[df["review_total_calc"] == 0, "positive_ratio_calc"] = pd.NA
    df["has_reviews_calc"] = df["review_total_calc"] > 0

    valid = df["has_reviews_calc"]
    reviewed = df.loc[valid].copy()

    ratio_p75 = reviewed["positive_ratio_calc"].quantile(0.75)
    reviews_p75 = reviewed["review_total_calc"].quantile(0.75)

    definitions = {
        "option_1_simple": valid & (df["positive_ratio_calc"] >= 0.90),
        "option_2_balanced": valid
        & (df["positive_ratio_calc"] >= 0.80)
        & (df["review_total_calc"] >= 50),
        "option_3_relative": valid
        & (df["positive_ratio_calc"] >= ratio_p75)
        & (df["review_total_calc"] >= reviews_p75),
    }

    comparison_rows = []
    for option_name, mask in definitions.items():
        success_df = df.loc[mask].copy()
        comparison_rows.append(
            {
                "option": option_name,
                "successful_games": int(mask.sum()),
                "pct_of_dataset": round(mask.mean() * 100, 2),
                "min_reviews_included": int(success_df["review_total_calc"].min()),
                "median_reviews_included": int(success_df["review_total_calc"].median()),
                "median_positive_ratio": round(success_df["positive_ratio_calc"].median(), 4),
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    final_mask = definitions["option_2_balanced"]
    df["success"] = final_mask.astype(int)
    df.to_csv(OUTPUT_CSV, index=False)

    inconsistent_total = int((df["num_reviews_total"] != df["review_total_calc"]).sum())
    inconsistent_zero = int(((df["num_reviews_total"] > 0) & (df["review_total_calc"] == 0)).sum())

    report = f"""# Steam Success Metric Report

## Scope
This report uses the latest cleaned file: `games_march2025_cleaned.csv` ({len(df):,} games).

## Day 1 - Candidate Variables For Success
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

- Rows where `num_reviews_total != positive + negative`: {inconsistent_total:,}
- Rows where `num_reviews_total > 0` but `positive + negative = 0`: {inconsistent_zero:,}

## Day 2 - Candidate Success Definitions
### Option 1 (Simple)
Plain English: A game is successful if at least 90% of its reviews are positive.

Pseudo code:
`success = 1 if positive_ratio_calc >= 0.90 else 0`

### Option 2 (Balanced)
Plain English: A game is successful if at least 80% of its reviews are positive and it has at least 50 total reviews.

Pseudo code:
`success = 1 if positive_ratio_calc >= 0.80 and review_total_calc >= 50 else 0`

### Option 3 (Relative)
Plain English: A game is successful if it is in the top 25% of reviewed games for both sentiment and review volume.

Pseudo code:
`success = 1 if positive_ratio_calc >= {ratio_p75:.4f} and review_total_calc >= {int(reviews_p75)} else 0`

## Day 3 - Comparison Of Options
| Option | Successful Games | % of Dataset | Included Review Count | Interpretation |
|---|---:|---:|---|---|
| Option 1 (Simple) | {comparison.loc[comparison["option"] == "option_1_simple", "successful_games"].iloc[0]:,} | {comparison.loc[comparison["option"] == "option_1_simple", "pct_of_dataset"].iloc[0]:.2f}% | Min {comparison.loc[comparison["option"] == "option_1_simple", "min_reviews_included"].iloc[0]:,}, median {comparison.loc[comparison["option"] == "option_1_simple", "median_reviews_included"].iloc[0]:,} | Too lenient. It includes many games with only 1-3 reviews, so it over-labels tiny games as successful. |
| Option 2 (Balanced) | {comparison.loc[comparison["option"] == "option_2_balanced", "successful_games"].iloc[0]:,} | {comparison.loc[comparison["option"] == "option_2_balanced", "pct_of_dataset"].iloc[0]:.2f}% | Min {comparison.loc[comparison["option"] == "option_2_balanced", "min_reviews_included"].iloc[0]:,}, median {comparison.loc[comparison["option"] == "option_2_balanced", "median_reviews_included"].iloc[0]:,} | Best balance. It removes tiny low-evidence games while keeping a usable number of positives. |
| Option 3 (Relative) | {comparison.loc[comparison["option"] == "option_3_relative", "successful_games"].iloc[0]:,} | {comparison.loc[comparison["option"] == "option_3_relative", "pct_of_dataset"].iloc[0]:.2f}% | Min {comparison.loc[comparison["option"] == "option_3_relative", "min_reviews_included"].iloc[0]:,}, median {comparison.loc[comparison["option"] == "option_3_relative", "median_reviews_included"].iloc[0]:,} | Too strict for a binary target. It keeps only an elite slice of games and risks class imbalance. |

## Day 4 - Final Definition
Selected definition: **Option 2 (Balanced)**

Final formula:

`success = 1 if positive_ratio_calc >= 0.80 and review_total_calc >= 50 else 0`

Why this works:

- It captures quality through strong sentiment.
- It captures popularity through minimum review evidence.
- It avoids the biggest failure of Option 1, which is labeling one-review games as successful.
- It avoids the biggest failure of Option 3, which is making success so exclusive that the positive class becomes too small.

Why the others were rejected:

- Option 1 was rejected because it is too lenient and includes many low-evidence games.
- Option 3 was rejected because it is too strict and leaves a relatively small positive class.

## Day 5 - Implemented Success Variable
The output file includes:

- `review_total_calc`
- `positive_ratio_calc`
- `has_reviews_calc`
- `success`

Class balance in the final output:

- Successful games: {int(final_mask.sum()):,}
- Unsuccessful games: {int((~final_mask).sum()):,}
- Positive class share: {final_mask.mean() * 100:.2f}%

## Day 6-7 - Documentation Write-Up
### Final definition
Success is defined as having at least 80% positive reviews and at least 50 total reviews, where total reviews are calculated as `positive + negative`.

### Reasoning
This definition balances product quality and market traction. A high positive ratio alone is not enough, because very small games can reach 100% positive with only a handful of reviews. Adding a minimum review threshold makes the label more credible while preserving enough successful games for downstream analysis or modeling.

### Limitations
1. Review volume is only one form of popularity; some games may be commercially successful with fewer Steam reviews.
2. The 50-review threshold is still somewhat judgment-based, even though it improves on the simple ratio-only rule.
3. The source file contains inconsistent precomputed review fields, so the analysis relies on `positive` and `negative` instead.
4. This is a Steam-platform success proxy, not a complete measure of revenue or long-term impact.
"""

    OUTPUT_MD.write_text(report)

    print(f"Saved report to: {OUTPUT_MD}")
    print(f"Saved labeled dataset to: {OUTPUT_CSV}")
    print("")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
