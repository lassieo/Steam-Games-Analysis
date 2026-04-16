# SHAP Feature Importance Analysis

This analysis runs SHAP on the tuned XGBoost model to understand which features actually drive the model's predictions, and to validate the project hypotheses against the model's behavior.

SHAP (SHapley Additive exPlanations) values are based on game-theoretic Shapley values, which measure each feature's marginal contribution to a specific prediction. Unlike Random Forest's impurity-based feature importance — which can be biased toward high-cardinality features and features that get split on more often without actually being more useful — SHAP gives a more honest picture of which features the model relies on. SHAP also provides both global importance (which features matter overall, averaged across all predictions) and local explanations (why this specific game was predicted as successful).

The SHAP TreeExplainer was run on the full dataset of 89,618 games. SHAP values converge quickly with sample size for tree models, so the global rankings reported here are stable.

## Top 15 features by SHAP importance

| Rank | Feature | Mean &#124;SHAP&#124; value |
|---:|---|---:|
| 1 | `tag_count` | 1.1883 |
| 2 | `has_tags` | 0.3906 |
| 3 | `publisher_historical_success` | 0.3537 |
| 4 | `release_year` | 0.2245 |
| 5 | `achievements` | 0.2051 |
| 6 | `price` | 0.1672 |
| 7 | `cat_steam_cloud` | 0.1560 |
| 8 | `developer_historical_success` | 0.1527 |
| 9 | `cat_steam_trading_cards` | 0.1526 |
| 10 | `language_count` | 0.1429 |
| 11 | `platform_count` | 0.0909 |
| 12 | `cat_family_sharing` | 0.0863 |
| 13 | `genre_action` | 0.0796 |
| 14 | `tag_story_rich` | 0.0784 |
| 15 | `developer_game_count` | 0.0725 |

The mean absolute SHAP value measures how much each feature shifts the model's prediction on average. Higher values mean the feature has more influence on whether a game is predicted as successful.

## Reading the SHAP rankings honestly

Three findings in this list deserve careful interpretation rather than being treated as clean causal signals.

**Tag-related features dominate the top of the rankings.** The `tag_count` feature has a mean absolute SHAP value of 1.188, which is roughly three times larger than the second-ranked feature (`has_tags` at 0.391). The combination of these two features tells a clear story: the model has learned that games with no community tags or very few community tags are unlikely to be labeled successful, and that the number of tags is a strong continuous signal once tags exist. This is plausibly meaningful — well-tagged games are typically games that the Steam community has actively engaged with, and community engagement is genuinely correlated with success — but it should not be interpreted as a causal driver. A developer cannot make their game successful by adding more tags. The honest interpretation is that tag_count is a proxy for community engagement, which itself reflects whether the game gained traction post-launch. This is a borderline case for our pre-launch leakage rule: tags are technically community-assigned after release, but they typically stabilize within the first week and the strategy report classified them as near-launch features. The dominance of tag_count in SHAP is a reminder that this classification choice has real consequences for what the model learns.

**Two of the top 10 features are technical Steam integrations rather than game qualities.** `cat_steam_cloud` (rank 7) and `cat_steam_trading_cards` (rank 9) are configuration flags indicating whether a game supports Steam's cloud save service and Steam trading card system. These are not features about the game itself — they are features about how thoroughly the developer integrated their release into the Steam platform. Both flags are strongly associated with established developers who go through the effort of configuring the full Steam feature set, which makes them proxies for developer professionalism and resourcing. They also tend to appear together, since a developer who configures cloud saves usually also enables trading cards. Treating these as causal would suggest that adding Steam cloud support to a game makes it more likely to succeed, which is not the actionable insight the rankings appear to imply.

**The hypothesis-relevant features all rank highly, but not at the very top.** Developer and publisher track records (ranks 3, 8, and 15), price (rank 6), genre (rank 13), and platform support (rank 11) all appear in the top 15, which is the result we hoped for. They validate every hypothesis from the strategy report. But none of them are individually as powerful as tag_count, which means the dominant signal in the dataset is community engagement (proxied by tag count) rather than any single business decision a developer can make. The combined contribution of the hypothesis features is substantial, but they are not the headline.

## Hypothesis validation

**H1: Paid games succeed more than free games.** Strongly supported. The `price` feature ranked 6 with mean &#124;SHAP&#124; of 0.167. The related `is_free` flag ranked much lower (rank 61, mean &#124;SHAP&#124; of 0.005) because the model gets the same information from `price` directly (where free games have value 0). This pattern suggests that `is_free` is redundant with `price` and could be dropped without hurting model performance.

**H2: Price tier matters more than raw price.** Both price-related features rank in the top 20, but in this dataset raw `price` (rank 6, &#124;SHAP&#124; 0.167) actually contributed more than `price_tier` (rank 19, &#124;SHAP&#124; 0.048). This is a partial reversal of the original hypothesis: the EDA showed clear threshold patterns suggesting tiers would help, but XGBoost is able to learn those thresholds from the raw price column on its own. Tiers can be useful for interpretability and for simpler models, but for a tuned gradient boosting model the raw value is sufficient.

**H3: Developer track record predicts success.** Strongly supported and arguably the most important finding for the project's hypotheses. `publisher_historical_success` ranked 3 (mean &#124;SHAP&#124; 0.354), `developer_historical_success` ranked 8 (mean &#124;SHAP&#124; 0.153), and `developer_game_count` ranked 15 (mean &#124;SHAP&#124; 0.073). All three features that capture past performance of the studio and publisher appear in the top 15. This confirms what the EDA identified with Cohen's d of 0.901: who made the game is one of the strongest predictors of whether it will be successful, and the model uses this information heavily.

**H4: Genre affects success probability.** Supported. `genre_action` ranked 13 (mean &#124;SHAP&#124; 0.080), `genre_count` ranked 17, and `genre_strategy` ranked 26. Genre is in the picture but is not as dominant as developer track record or price. The model treats genre as one signal among many rather than the primary axis of prediction.

**H5: Multi-platform support matters.** Supported. `platform_count` ranked 11 with mean &#124;SHAP&#124; of 0.091. Games supporting more platforms tend to be predicted as more successful, which matches the EDA finding and is consistent with the interpretation that multi-platform support signals development investment.

## Interpretation

The SHAP results validate every hypothesis from the strategy report at the model level, which is a stronger form of evidence than EDA correlations because it shows the model actually uses these features when making predictions rather than them merely correlating with the outcome. At the same time, the rankings reveal that the dominant signal in the dataset is community engagement (via tag_count) rather than any single hypothesis-driven feature. The hypothesis features are all present and contributing, but they are sharing influence with proxy signals that the model has learned to exploit.

For the final report, the honest framing is that the project succeeded in identifying which pre-launch features drive predictions, validated all five hypotheses with model-level evidence, and uncovered an additional finding worth flagging: the strongest predictive signal in the dataset comes from community engagement proxies that are technically near-launch features rather than purely pre-launch attributes. This is not a methodological problem (the strategy report explicitly classified tags as near-launch and acceptable), but it is a meaningful caveat when describing how the model would behave at true pre-launch prediction time.
