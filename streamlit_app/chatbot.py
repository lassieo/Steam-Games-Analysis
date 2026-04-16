"""
Chatbot Tab
============

Natural language Q&A about the Steam games dataset. Uses a lightweight
RAG pattern: the user's question is matched to a data query, the query
runs against the engineered CSV, and the resulting numbers are passed
to Groq's Llama model with a prompt that forces data-grounded answers.

Supports:
  - Aggregate stats (genre, price, developer, platform, year, etc.)
  - Individual game lookups by name (with or without quotes)
  - Top-N queries with composable filters (genre, year, free/paid)
  - Recommendation requests ("suggest", "recommend", "any games")

No vector database, no embeddings. The data is structured, so we query
it directly. This is faster, more accurate, and simpler than a full
RAG stack for a project of this size.
"""
from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
import streamlit as st

from app_utils import load_data


SYSTEM_PROMPT = """You are a data analyst assistant for a Steam games success prediction project. You have access to real numbers computed from a dataset of 89,618 games, and you must answer user questions using ONLY the data provided in the context.

Rules:
- Never invent numbers or game names. If the context doesn't contain specific game names, say so honestly.
- When the context provides a list of games, present them clearly with names, years, and prices.
- When discussing individual games, use the exact data provided in the context.
- Keep answers conversational but precise. Include specific numbers and percentages when they're in the context.
- If the user asks about the modeling approach, mention that we used a Random Forest baseline, XGBoost (tuned with Optuna), LightGBM, MLP, and a stacking ensemble, with the stacking ensemble achieving the best ROC-AUC of 0.884.
- If the user asks about hypothesis validation, mention that all 5 hypotheses were validated by SHAP analysis, with developer/publisher track record being the strongest predictor.

Answer in 2-4 sentences unless presenting a list of games, in which case format the list clearly."""


# All the genre features in the dataset (lowercase keys we'll match against)
KNOWN_GENRES = {
    "action": "genre_action",
    "adventure": "genre_adventure",
    "casual": "genre_casual",
    "indie": "genre_indie",
    "rpg": "genre_rpg",
    "simulation": "genre_simulation",
    "strategy": "genre_strategy",
    "sports": "genre_sports",
    "racing": "genre_racing",
    "horror": "genre_horror",  # may or may not exist as a top-15 column
    "free to play": "genre_free_to_play",
    "free_to_play": "genre_free_to_play",
    "freetoplay": "genre_free_to_play",
    "early access": "genre_early_access",
    "massively multiplayer": "genre_massively_multiplayer",
    "mmo": "genre_massively_multiplayer",
}

# Words that signal the user wants a list of recommendations or top items
TOP_N_TRIGGERS = [
    "top", "best", "most successful", "highest", "leading",
    "suggest", "recommend", "recommendation", "any games", "show me", "list",
    "find me", "give me",
]


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
def detect_top_n(question: str) -> int | None:
    """
    Detect if the user is asking for top-N or any recommendation-style query.
    Returns the number of items to return (defaults to 5 if no specific number).
    """
    q = question.lower()
    if any(trigger in q for trigger in TOP_N_TRIGGERS):
        match = re.search(r"top\s+(\d+)", q)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s+(?:games|titles)", q)
        if match:
            n = int(match.group(1))
            return min(n, 20)  # cap at 20
        return 5
    return None


def detect_year_filter(question: str) -> int | None:
    """Detect if the user is asking about a specific year."""
    match = re.search(r"\b(19[89]\d|20[0-3]\d)\b", question)
    if match:
        return int(match.group(1))
    return None


def detect_genres(question: str, df: pd.DataFrame) -> list[str]:
    """
    Detect which genre column(s) the user is asking about.
    Returns a list of column names that exist in the dataframe.
    """
    q = question.lower()
    detected = []
    available_columns = set(df.columns)

    for keyword, col_name in KNOWN_GENRES.items():
        if keyword in q and col_name in available_columns and col_name not in detected:
            detected.append(col_name)

    return detected


def detect_free_paid_filter(question: str) -> bool | None:
    """Returns True for free-only, False for paid-only, None for no filter."""
    q = question.lower()
    if "free" in q and "to play" in q:
        return True
    if any(w in q for w in [" free ", "free games", "free game", "f2p", "free-to-play"]):
        return True
    if any(w in q for w in ["paid", "premium games"]):
        return False
    return None


def detect_game_lookup(question: str, df: pd.DataFrame) -> str | None:
    """
    Detect if the user is asking about a specific game by name.
    Tries multiple strategies in order of confidence:
      1. Quoted strings (most reliable)
      2. After 'about/for/of' prefix
      3. Substring search of the whole question against game names
    """
    # Strategy 1: quoted strings
    quoted = re.findall(r'"([^"]+)"', question) + re.findall(r"'([^']+)'", question)
    for q in quoted:
        match = _find_game_by_name(df, q)
        if match:
            return match

    # Strategy 2: text after "about/for/of/like/play"
    name_match = re.search(
        r"(?:about|for|of|like|play|liked|played|enjoyed)\s+([A-Za-z0-9'\-:!?\s]{3,60})",
        question,
        re.IGNORECASE,
    )
    if name_match:
        candidate = name_match.group(1).strip().rstrip(".?!,")
        # Try the whole candidate first, then progressively shorter
        for n_words in range(min(8, len(candidate.split())), 1, -1):
            short = " ".join(candidate.split()[:n_words])
            match = _find_game_by_name(df, short)
            if match:
                return match

    # Strategy 3: try the whole question (last resort, only if it looks like a title)
    # Skip if it's clearly a question word
    question_starters = ["what", "which", "how", "why", "when", "where", "do ", "does", "is ", "are ", "can ", "show", "list", "top", "suggest", "recommend"]
    q_lower = question.lower().strip()
    if not any(q_lower.startswith(w) for w in question_starters):
        # Try the whole thing as a game name
        clean = re.sub(r"[?!.,]", "", question).strip()
        match = _find_game_by_name(df, clean)
        if match:
            return match

    return None


def _find_game_by_name(df: pd.DataFrame, name: str) -> str | None:
    """Try exact match first, then case-insensitive substring."""
    if not name or len(name) < 3:
        return None
    name = name.strip()

    # Exact match
    matches = df[df["name"].str.lower() == name.lower()]
    if len(matches) > 0:
        return matches.iloc[0]["name"]

    # Substring match (must be at least 4 chars to avoid false positives)
    if len(name) >= 4:
        try:
            matches = df[df["name"].str.lower().str.contains(re.escape(name.lower()), na=False)]
            if len(matches) > 0:
                # Prefer shorter names (more likely to be the actual game)
                return matches.sort_values("name", key=lambda s: s.str.len()).iloc[0]["name"]
        except Exception:
            pass

    return None


def analyze_question_intent(question: str, df: pd.DataFrame) -> dict:
    """Analyze the question and return all detected intents and parameters."""
    q = question.lower()
    intents = {
        "categories": [],
        "top_n": detect_top_n(question),
        "year_filter": detect_year_filter(question),
        "game_name": None,
        "genre_filters": detect_genres(question, df),
        "free_filter": detect_free_paid_filter(question),
    }

    # Only try game lookup if it's not clearly a top-N query
    if intents["top_n"] is None:
        intents["game_name"] = detect_game_lookup(question, df)

    if intents["game_name"]:
        intents["categories"].append("game_lookup")

    if intents["top_n"] is not None:
        intents["categories"].append("top_n")

    if intents["genre_filters"]:
        intents["categories"].append("genre")
    elif any(w in q for w in ["genre"]):
        intents["categories"].append("genre")

    if any(w in q for w in ["price", "cost", "free", "paid", "expensive", "cheap", "budget", "premium"]):
        intents["categories"].append("price")
    if any(w in q for w in ["developer", "publisher", "studio", "track record", "experience"]):
        intents["categories"].append("developer")
    if any(w in q for w in ["platform", "windows", "mac", "linux", "multi-platform"]):
        intents["categories"].append("platform")
    if any(w in q for w in ["year", "release", "trend", "temporal", "time", "when"]):
        intents["categories"].append("year")
    if any(w in q for w in ["success rate", "percent", "how many", "total", "count"]):
        intents["categories"].append("overall")
    if any(w in q for w in ["model", "accuracy", "performance", "roc", "auc", "f1", "predict"]):
        intents["categories"].append("model")
    if any(w in q for w in ["feature", "important", "shap", "matter", "driver", "predictor"]):
        intents["categories"].append("features")
    if any(w in q for w in ["language", "localization", "translate"]):
        intents["categories"].append("language")
    if any(w in q for w in [" tag ", " tags", "category"]):
        intents["categories"].append("tags")

    if not intents["categories"]:
        intents["categories"].append("overall")

    return intents


# ---------------------------------------------------------------------------
# Data query functions
# ---------------------------------------------------------------------------
def query_individual_game(df: pd.DataFrame, game_name: str) -> str:
    """Pull all relevant info about a single game."""
    matches = df[df["name"] == game_name]
    if len(matches) == 0:
        return f"No game found matching '{game_name}'."

    game = matches.iloc[0]
    success_status = "successful" if game["success"] == 1 else "not successful"

    info = [f"Game: **{game['name']}**"]
    info.append(f"- Success status: **{success_status}** (per our 80% positive + 50 reviews definition)")

    if "release_year" in game and pd.notna(game["release_year"]):
        info.append(f"- Release year: {int(game['release_year'])}")
    if "price" in game and pd.notna(game["price"]):
        price_text = "Free" if game["price"] == 0 else f"${game['price']:.2f}"
        info.append(f"- Price: {price_text}")
    if "developer_historical_success" in game:
        info.append(f"- Developer historical success rate: {game['developer_historical_success']:.2f}")
    if "publisher_historical_success" in game:
        info.append(f"- Publisher historical success rate: {game['publisher_historical_success']:.2f}")
    if "tag_count" in game:
        info.append(f"- Number of community tags: {int(game['tag_count'])}")
    if "achievements" in game:
        info.append(f"- Number of achievements: {int(game['achievements'])}")
    if "supported_language_count" in game:
        info.append(f"- Supported languages: {int(game['supported_language_count'])}")

    platforms = []
    for p in ["windows", "mac", "linux"]:
        if p in game and game[p] == 1:
            platforms.append(p.title())
    if platforms:
        info.append(f"- Platforms: {', '.join(platforms)}")

    genre_cols = [c for c in df.columns if c.startswith("genre_") and c != "genre_count"]
    genres = [c.replace("genre_", "").replace("_", " ").title() for c in genre_cols if game.get(c) == 1]
    if genres:
        info.append(f"- Genres: {', '.join(genres)}")

    return "\n".join(info)


def query_top_n_games(
    df: pd.DataFrame,
    n: int,
    genre_filters: list[str] = None,
    year_filter: int | None = None,
    free_filter: bool | None = None,
) -> str:
    """
    Get top-N successful games with composable filters.
    Sorted by tag_count (engagement proxy) since we don't have raw review counts.
    """
    filtered = df.copy()
    filter_desc = []

    # Apply genre filters (AND logic — game must be in ALL specified genres)
    if genre_filters:
        for col in genre_filters:
            if col in filtered.columns:
                filtered = filtered[filtered[col] == 1]
                genre_label = col.replace("genre_", "").replace("_", " ").title()
                filter_desc.append(genre_label)

    # Year filter
    if year_filter:
        filtered = filtered[filtered["release_year"] == year_filter]
        filter_desc.append(f"released in {year_filter}")

    # Free/paid filter
    if free_filter is True:
        filtered = filtered[filtered["price"] == 0]
        filter_desc.append("free-to-play")
    elif free_filter is False:
        filtered = filtered[filtered["price"] > 0]
        filter_desc.append("paid")

    # Filter to successful games only
    successful = filtered[filtered["success"] == 1].copy()

    if len(successful) == 0:
        criteria = " + ".join(filter_desc) if filter_desc else "the dataset"
        return (
            f"No successful games found matching: {criteria}\n"
            f"(Total games matching filters but not successful: {len(filtered):,})"
        )

    # Sort by tag_count (engagement proxy)
    if "tag_count" in successful.columns:
        successful = successful.sort_values("tag_count", ascending=False)

    top = successful.head(n)
    criteria = " + ".join(filter_desc) if filter_desc else "all categories"

    lines = [
        f"Top {len(top)} successful games matching: **{criteria}**",
        f"(Filtered from {len(successful):,} total successful games matching the criteria. "
        f"Sorted by community tag count as a proxy for engagement.)",
        "",
    ]
    for i, (_, row) in enumerate(top.iterrows(), 1):
        price = "Free" if row["price"] == 0 else f"${row['price']:.2f}"
        tags = int(row.get("tag_count", 0))
        year = int(row.get("release_year", 0)) if pd.notna(row.get("release_year")) else "?"
        lines.append(f"{i}. **{row['name']}** ({year}) - {price}, {tags} community tags")

    return "\n".join(lines)


def query_data(df: pd.DataFrame, intents: dict) -> str:
    """Pull numbers from the dataframe based on the detected intents."""
    blocks = []
    categories = intents.get("categories", [])

    # Always include overall stats
    total = len(df)
    successful = int(df["success"].sum())
    success_rate = successful / total

    blocks.append(
        f"Overall dataset: {total:,} games, {successful:,} successful "
        f"({success_rate * 100:.1f}% positive class). Success defined as "
        f"positive_ratio >= 80% AND total reviews >= 50."
    )

    # --- Individual game lookup ---
    if "game_lookup" in categories and intents.get("game_name"):
        blocks.append(query_individual_game(df, intents["game_name"]))

    # --- Top-N query ---
    if "top_n" in categories:
        n = intents.get("top_n", 5)
        result = query_top_n_games(
            df,
            n=n,
            genre_filters=intents.get("genre_filters", []),
            year_filter=intents.get("year_filter"),
            free_filter=intents.get("free_filter"),
        )
        blocks.append(result)

    # --- Genre stats (only if not already a top-N query, otherwise it's redundant) ---
    if "genre" in categories and "top_n" not in categories:
        genre_cols = [c for c in df.columns if c.startswith("genre_") and c != "genre_count"]
        if genre_cols:
            genre_stats = []
            for col in genre_cols[:15]:
                mask = df[col] == 1
                if mask.sum() >= 500:
                    sr = df.loc[mask, "success"].mean()
                    genre_stats.append((col.replace("genre_", "").replace("_", " ").title(), int(mask.sum()), sr))
            genre_stats.sort(key=lambda t: t[2], reverse=True)
            if genre_stats:
                lines = [f"- {name}: {n:,} games, {sr * 100:.1f}% success rate"
                         for name, n, sr in genre_stats]
                blocks.append("Success rate by genre:\n" + "\n".join(lines))

    # --- Price stats ---
    if "price" in categories and "top_n" not in categories:
        free_mask = df["price"] == 0
        paid_mask = df["price"] > 0
        if free_mask.sum() > 0 and paid_mask.sum() > 0:
            free_sr = df.loc[free_mask, "success"].mean()
            paid_sr = df.loc[paid_mask, "success"].mean()
            blocks.append(
                f"Free vs paid: {int(free_mask.sum()):,} free games have a "
                f"{free_sr * 100:.1f}% success rate, while {int(paid_mask.sum()):,} "
                f"paid games have a {paid_sr * 100:.1f}% success rate."
            )

        tiers = [
            ("Budget (<$15)", (df["price"] > 0) & (df["price"] < 15)),
            ("Mid ($15-50)", (df["price"] >= 15) & (df["price"] < 50)),
            ("Premium ($50-85)", (df["price"] >= 50) & (df["price"] < 85)),
            ("AAA ($85+)", df["price"] >= 85),
        ]
        lines = []
        for label, mask in tiers:
            if mask.sum() > 0:
                lines.append(
                    f"- {label}: {int(mask.sum()):,} games, "
                    f"{df.loc[mask, 'success'].mean() * 100:.1f}% success rate"
                )
        if lines:
            blocks.append("Success rate by price tier:\n" + "\n".join(lines))

    # --- Developer stats ---
    if "developer" in categories:
        if "developer_historical_success" in df.columns:
            no_hist = df["developer_historical_success"] == 0
            some_hist = (df["developer_historical_success"] > 0) & (df["developer_historical_success"] < 0.5)
            strong_hist = df["developer_historical_success"] >= 0.5
            lines = []
            for label, mask in [
                ("No prior success (new or struggling devs)", no_hist),
                ("Some prior success (0-50%)", some_hist),
                ("Strong track record (50%+)", strong_hist),
            ]:
                if mask.sum() > 0:
                    lines.append(
                        f"- {label}: {int(mask.sum()):,} games, "
                        f"{df.loc[mask, 'success'].mean() * 100:.1f}% success rate"
                    )
            if lines:
                blocks.append("Success rate by developer track record:\n" + "\n".join(lines))

        blocks.append(
            "Our EDA found developer_historical_success has a Cohen's d of 0.901, "
            "which is the strongest class separation of any feature in the dataset."
        )

    # --- Platform stats ---
    if "platform" in categories:
        if all(c in df.columns for c in ["windows", "mac", "linux"]):
            df_tmp = df.copy()
            df_tmp["_platform_count"] = df_tmp["windows"] + df_tmp["mac"] + df_tmp["linux"]
            lines = []
            for n_p in [1, 2, 3]:
                mask = df_tmp["_platform_count"] == n_p
                if mask.sum() > 0:
                    lines.append(
                        f"- {n_p} platform(s): {int(mask.sum()):,} games, "
                        f"{df_tmp.loc[mask, 'success'].mean() * 100:.1f}% success rate"
                    )
            if lines:
                blocks.append("Success rate by platform count:\n" + "\n".join(lines))

    # --- Year stats ---
    if "year" in categories and "top_n" not in categories:
        if "release_year" in df.columns:
            year_filter = intents.get("year_filter")
            if year_filter:
                year_data = df[df["release_year"] == year_filter]
                if len(year_data) > 0:
                    blocks.append(
                        f"Year {year_filter}: {len(year_data):,} games, "
                        f"{int(year_data['success'].sum()):,} successful "
                        f"({year_data['success'].mean() * 100:.1f}% success rate)."
                    )
            else:
                year_stats = df.groupby("release_year")["success"].agg(["count", "mean"]).reset_index()
                year_stats = year_stats[year_stats["count"] >= 200].tail(10)
                lines = [
                    f"- {int(row['release_year'])}: {int(row['count']):,} games, "
                    f"{row['mean'] * 100:.1f}% success rate"
                    for _, row in year_stats.iterrows()
                ]
                if lines:
                    blocks.append("Success rate by release year (recent):\n" + "\n".join(lines))

    # --- Language stats ---
    if "language" in categories:
        if "supported_language_count" in df.columns:
            lang_groups = [
                ("1 language (English only)", df["supported_language_count"] == 1),
                ("2-5 languages", (df["supported_language_count"] >= 2) & (df["supported_language_count"] <= 5)),
                ("6-10 languages", (df["supported_language_count"] >= 6) & (df["supported_language_count"] <= 10)),
                ("10+ languages", df["supported_language_count"] > 10),
            ]
            lines = []
            for label, mask in lang_groups:
                if mask.sum() > 0:
                    lines.append(
                        f"- {label}: {int(mask.sum()):,} games, "
                        f"{df.loc[mask, 'success'].mean() * 100:.1f}% success rate"
                    )
            if lines:
                blocks.append("Success rate by language support:\n" + "\n".join(lines))

    # --- Tag stats ---
    if "tags" in categories:
        if "tag_count" in df.columns:
            no_tags = df["tag_count"] == 0
            few_tags = (df["tag_count"] > 0) & (df["tag_count"] <= 10)
            many_tags = df["tag_count"] > 10
            lines = []
            for label, mask in [
                ("No tags", no_tags),
                ("1-10 tags", few_tags),
                ("11+ tags", many_tags),
            ]:
                if mask.sum() > 0:
                    lines.append(
                        f"- {label}: {int(mask.sum()):,} games, "
                        f"{df.loc[mask, 'success'].mean() * 100:.1f}% success rate"
                    )
            if lines:
                blocks.append("Success rate by tag count:\n" + "\n".join(lines))

    # --- Model performance ---
    if "model" in categories:
        blocks.append(
            "Model comparison results (5-fold stratified cross-validation, all using the same splits):\n"
            "- Stacking Ensemble: ROC-AUC 0.8844, F1 0.568 (best)\n"
            "- XGBoost (Optuna-tuned): ROC-AUC 0.8841, F1 0.576\n"
            "- LightGBM: ROC-AUC 0.8799, F1 0.556\n"
            "- Random Forest baseline: ROC-AUC 0.8792, F1 0.434\n"
            "- XGBoost (untuned): ROC-AUC 0.8748, F1 0.564\n"
            "- Tabular MLP: ROC-AUC 0.8724, F1 0.488"
        )

    # --- SHAP features ---
    if "features" in categories:
        blocks.append(
            "Top features by SHAP importance (from tuned XGBoost):\n"
            "1. tag_count (mean |SHAP| = 1.188) - community engagement proxy\n"
            "2. has_tags (0.391)\n"
            "3. publisher_historical_success (0.354)\n"
            "4. release_year (0.225)\n"
            "5. achievements (0.205)\n"
            "6. price (0.167)\n"
            "7. cat_steam_cloud (0.156)\n"
            "8. developer_historical_success (0.153)\n"
            "9. cat_steam_trading_cards (0.153)\n"
            "10. language_count (0.143)"
        )

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Groq API call
# ---------------------------------------------------------------------------
def get_groq_client():
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return "missing_package"

    return Groq(api_key=api_key)


def ask_llm(client, user_question: str, data_context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Data context:\n{data_context}\n\nQuestion: {user_question}"},
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=700,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main tab render
# ---------------------------------------------------------------------------
def render_chatbot_tab() -> None:
    st.header("💬 Ask the Data")
    st.markdown(
        "Ask questions about Steam games in natural language. The chatbot queries "
        "the real dataset to get exact numbers, then uses Groq's Llama 3.3 70B to "
        "turn them into a conversational answer."
    )

    with st.expander("ℹ️ What can this chatbot answer?", expanded=False):
        st.markdown(
            """
            This chatbot is **data-grounded** — every answer comes from a real query against the Steam dataset.

            **Aggregate statistics**
            - Success rates by genre, price tier, developer track record, platform, year
            - Comparisons (free vs paid, multi-platform vs single platform)

            **Individual game lookups**
            - "Tell me about Stardew Valley" or `What is "Hades"?`

            **Top-N and recommendations** (filters can be combined)
            - "Top 5 successful indie games"
            - "Suggest free-to-play games that are also action"
            - "Best horror games released in 2023"
            - "Recommend any RPG games"

            **Project context**
            - Model performance, SHAP feature importance, hypothesis validation

            **What it can't answer:**
            - Future predictions (use the Predictor tab)
            - Information not in the dataset (lore, gameplay mechanics, current player counts)
            """
        )

    df = load_data()
    client = get_groq_client()

    if client == "missing_package":
        st.error("The `groq` package isn't installed. Add it to requirements.txt and redeploy.")
        return

    if client is None:
        st.warning(
            "Groq API key not configured. The chatbot won't work until you add your key. "
            "Locally: set `GROQ_API_KEY` environment variable. "
            "On Streamlit Cloud: add it via app settings → Secrets."
        )
        return

    st.subheader("Try these questions")
    sample_cols = st.columns(2)
    samples = [
        "Which genre has the highest success rate?",
        "Do paid games do better than free games?",
        'Tell me about "Stardew Valley"',
        "Top 5 successful indie games",
        "Suggest free-to-play action games",
        "Best games from 2023",
        "Recommend any horror games",
        "How did the stacking ensemble perform?",
    ]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for i, sample in enumerate(samples):
        col = sample_cols[i % 2]
        if col.button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = sample

    st.markdown("---")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask a question about the data...") or pending

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Querying data and asking Groq..."):
                intents = analyze_question_intent(user_input, df)
                data_context = query_data(df, intents)
                try:
                    answer = ask_llm(client, user_input, data_context)
                except Exception as exc:
                    answer = (
                        f"Sorry, the Groq API call failed: {exc}\n\n"
                        f"Here's the raw data I found:\n\n{data_context}"
                    )
                st.markdown(answer)

                # Show what intents were detected
                detected_categories = ", ".join(intents["categories"]) if intents["categories"] else "none"
                with st.expander(f"🔍 Detected intents: {detected_categories}"):
                    if intents.get("game_name"):
                        st.write(f"**Game lookup:** {intents['game_name']}")
                    if intents.get("top_n"):
                        st.write(f"**Top N:** {intents['top_n']}")
                    if intents.get("year_filter"):
                        st.write(f"**Year filter:** {intents['year_filter']}")
                    if intents.get("genre_filters"):
                        st.write(f"**Genre filters:** {', '.join(intents['genre_filters'])}")
                    if intents.get("free_filter") is not None:
                        st.write(f"**Free/paid filter:** {'free only' if intents['free_filter'] else 'paid only'}")
                    st.markdown("**Raw data context passed to LLM:**")
                    st.text(data_context)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()