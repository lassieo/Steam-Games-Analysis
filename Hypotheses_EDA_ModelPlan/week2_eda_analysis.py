from pathlib import Path

import pandas as pd


BASE_DIR = Path("/Users/laasyavenugopal/Desktop/Steam Datasets/Week2_Steam")
SOURCE_FILE = Path("/Users/laasyavenugopal/Desktop/Steam Datasets/games_march2025_with_success.csv")


def main() -> None:
    df = pd.read_csv(
        SOURCE_FILE,
        usecols=["success", "review_total_calc", "positive_ratio_calc", "price"],
        low_memory=False,
    )

    review_bins = [-1, 0, 4, 9, 24, 49, 99, 499, 999, 10**12]
    review_labels = ["0", "1-4", "5-9", "10-24", "25-49", "50-99", "100-499", "500-999", "1000+"]
    df["review_bin"] = pd.cut(df["review_total_calc"], bins=review_bins, labels=review_labels)

    review_summary = (
        df.groupby("review_bin", observed=False)
        .agg(
            games=("success", "size"),
            success_rate=("success", "mean"),
            median_reviews=("review_total_calc", "median"),
            median_positive_ratio=("positive_ratio_calc", "median"),
        )
        .reset_index()
    )
    review_summary["success_rate_pct"] = (review_summary["success_rate"] * 100).round(2)
    review_summary.to_csv(BASE_DIR / "review_hypothesis_summary.csv", index=False)

    price_bins = [-0.01, 0, 4.99, 9.99, 19.99, 29.99, 59.99, 9999]
    price_labels = ["Free", "0.01-4.99", "5.00-9.99", "10.00-19.99", "20.00-29.99", "30.00-59.99", "60+"]
    df["price_bin"] = pd.cut(df["price"], bins=price_bins, labels=price_labels)

    price_summary = (
        df.groupby("price_bin", observed=False)
        .agg(
            games=("success", "size"),
            success_rate=("success", "mean"),
            median_reviews=("review_total_calc", "median"),
            median_positive_ratio=("positive_ratio_calc", "median"),
            median_price=("price", "median"),
        )
        .reset_index()
    )
    price_summary["success_rate_pct"] = (price_summary["success_rate"] * 100).round(2)
    price_summary.to_csv(BASE_DIR / "price_hypothesis_summary.csv", index=False)

    review_ready = df[df["review_total_calc"] >= 50].copy()
    review_ready_price_summary = (
        review_ready.groupby("price_bin", observed=False)
        .agg(
            games=("success", "size"),
            success_rate=("success", "mean"),
            median_reviews=("review_total_calc", "median"),
        )
        .reset_index()
    )
    review_ready_price_summary["success_rate_pct"] = (
        review_ready_price_summary["success_rate"] * 100
    ).round(2)
    review_ready_price_summary.to_csv(BASE_DIR / "price_summary_review_threshold.csv", index=False)

    print(f"Saved: {BASE_DIR / 'review_hypothesis_summary.csv'}")
    print(f"Saved: {BASE_DIR / 'price_hypothesis_summary.csv'}")
    print(f"Saved: {BASE_DIR / 'price_summary_review_threshold.csv'}")


if __name__ == "__main__":
    main()
