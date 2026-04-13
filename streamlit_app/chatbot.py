"""
Chatbot Tab
============

Natural language Q&A about the Steam games dataset. Uses a lightweight
RAG pattern: the user's question is matched to a data query, the query
runs against the engineered CSV, and the resulting numbers are passed
to Groq's Llama model with a prompt that forces data-grounded answers.

No vector database, no embeddings. The data is structured, so we query
it directly. This is faster, more accurate, and simpler than a full
RAG stack for a project of this size.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from app_utils import load_data


# Prompt that constrains the LLM to answer only using provided data
SYSTEM_PROMPT = """You are a data analyst assistant for a Steam games success prediction project. You have access to real numbers computed from a dataset of 89,618 games, and you must answer user questions using ONLY the data provided in the context.

Rules:
- Never invent numbers. If the context doesn't contain the answer, say so.
- Keep answers conversational but precise. Include specific numbers and percentages when they're in the context.
- When comparing things, cite the actual values.
- If the user asks about the modeling approach (not the data), mention that we used a Random Forest baseline, XGBoost (tuned with Optuna), LightGBM, MLP, and a stacking ensemble, with the stacking ensemble achieving the best ROC-AUC of 0.884.
- If the user asks about hypothesis validation, mention that all 5 hypotheses were validated by SHAP analysis, with developer/publisher track record being the strongest predictor.

Answer in 2-4 sentences unless the user asks for more detail."""


# ---------------------------------------------------------------------------
# Query routing — map question intent to data operations
# ---------------------------------------------------------------------------
def analyze_question_intent(question: str) -> list[str]:
    """
    Returns a list of data 'intents' the question touches.
    We use keyword matching because the intent space is small and predictable.
    """
    q = question.lower()
    intents = []

    if any(w in q for w in ["genre", "rpg", "action", "adventure", "indie", "strategy"]):
        intents.append("genre")
    if any(w in q for w in ["price", "cost", "free", "paid", "expensive", "cheap"]):
        intents.append("price")
    if any(w in q for w in ["developer", "publisher", "studio", "track record", "experience"]):
        intents.append("developer")
    if any(w in q for w in ["platform", "windows", "mac", "linux", "multi"]):
        intents.append("platform")
    if any(w in q for w in ["year", "release", "trend", "temporal", "time"]):
        intents.append("year")
    if any(w in q for w in ["success rate", "percent", "how many", "total", "overall", "count"]):
        intents.append("overall")
    if any(w in q for w in ["model", "accuracy", "performance", "roc", "auc", "f1", "predict"]):
        intents.append("model")
    if any(w in q for w in ["feature", "important", "shap", "matter", "driver", "predictor"]):
        intents.append("features")
    if any(w in q for w in ["language", "localization", "translate"]):
        intents.append("language")
    if any(w in q for w in ["tag", "category"]):
        intents.append("tags")

    if not intents:
        intents.append("overall")

    return intents


def query_data(df: pd.DataFrame, intents: list[str]) -> str:
    """
    Pull numbers from the dataframe based on the detected intents.
    Returns a formatted text block the LLM will use as context.
    """
    blocks = []

    # Always include overall stats
    total = len(df)
    successful = int(df["success"].sum())
    success_rate = successful / total

    blocks.append(
        f"Overall dataset: {total:,} games, {successful:,} successful "
        f"({success_rate * 100:.1f}% positive class). Success defined as "
        f"positive_ratio >= 80% AND total reviews >= 50."
    )

    if "genre" in intents:
        genre_cols = [c for c in df.columns if c.startswith("genre_") and c != "genre_count"]
        if genre_cols:
            genre_stats = []
            for col in genre_cols[:15]:
                mask = df[col] == 1
                if mask.sum() >= 500:  # min sample size
                    sr = df.loc[mask, "success"].mean()
                    genre_stats.append((col.replace("genre_", "").title(), int(mask.sum()), sr))
            genre_stats.sort(key=lambda t: t[2], reverse=True)
            if genre_stats:
                lines = [f"- {name}: {n:,} games, {sr * 100:.1f}% success rate"
                         for name, n, sr in genre_stats]
                blocks.append("Success rate by genre:\n" + "\n".join(lines))

    if "price" in intents:
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

        # Price tier breakdown if available
        tiers = [
            ("Under $5", (df["price"] > 0) & (df["price"] < 5)),
            ("$5-10", (df["price"] >= 5) & (df["price"] < 10)),
            ("$10-20", (df["price"] >= 10) & (df["price"] < 20)),
            ("$20-30", (df["price"] >= 20) & (df["price"] < 30)),
            ("$30+", df["price"] >= 30),
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

    if "developer" in intents:
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

    if "platform" in intents:
        if all(c in df.columns for c in ["windows", "mac", "linux"]):
            df_tmp = df.copy()
            df_tmp["_platform_count"] = df_tmp["windows"] + df_tmp["mac"] + df_tmp["linux"]
            lines = []
            for n in [1, 2, 3]:
                mask = df_tmp["_platform_count"] == n
                if mask.sum() > 0:
                    lines.append(
                        f"- {n} platform(s): {int(mask.sum()):,} games, "
                        f"{df_tmp.loc[mask, 'success'].mean() * 100:.1f}% success rate"
                    )
            if lines:
                blocks.append("Success rate by platform count:\n" + "\n".join(lines))

    if "year" in intents:
        if "release_year" in df.columns:
            year_stats = df.groupby("release_year")["success"].agg(["count", "mean"]).reset_index()
            year_stats = year_stats[year_stats["count"] >= 200].tail(10)
            lines = [
                f"- {int(row['release_year'])}: {int(row['count']):,} games, "
                f"{row['mean'] * 100:.1f}% success rate"
                for _, row in year_stats.iterrows()
            ]
            if lines:
                blocks.append("Success rate by release year (recent):\n" + "\n".join(lines))

    if "language" in intents:
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

    if "tags" in intents:
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

    if "model" in intents:
        blocks.append(
            "Model comparison results (5-fold stratified cross-validation, all using the same splits):\n"
            "- Stacking Ensemble: ROC-AUC 0.8844, F1 0.568 (best)\n"
            "- XGBoost (Optuna-tuned): ROC-AUC 0.8841\n"
            "- LightGBM: ROC-AUC 0.8799, F1 0.556\n"
            "- Random Forest baseline: ROC-AUC 0.8792, F1 0.434\n"
            "- XGBoost (untuned): ROC-AUC 0.8748, F1 0.564\n"
            "- Tabular MLP: ROC-AUC 0.8724, F1 0.488"
        )

    if "features" in intents:
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
    """Read the API key from Streamlit secrets or env variable."""
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
    """Send the question and data context to Groq and return the response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Data context:\n{data_context}\n\nQuestion: {user_question}"},
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main tab render
# ---------------------------------------------------------------------------
def render_chatbot_tab() -> None:
    st.header("Ask the Data")
    st.markdown(
        "Ask questions about Steam games in natural language. The chatbot queries "
        "the real dataset to get exact numbers, then uses Groq's Llama 3.3 70B to "
        "turn them into a conversational answer."
    )

    df = load_data()
    client = get_groq_client()

    if client == "missing_package":
        st.error(
            "The `groq` package isn't installed. Add it to requirements.txt and redeploy."
        )
        return

    if client is None:
        st.warning(
            "Groq API key not configured. The chatbot won't work until you add your key. "
            "Locally: set `GROQ_API_KEY` environment variable. "
            "On Streamlit Cloud: add it via app settings → Secrets."
        )
        st.caption("You can still use the suggested questions below to see the data queries.")
        return

    # Sample questions
    st.subheader("Try these questions")
    sample_cols = st.columns(2)
    samples = [
        "Which genre has the highest success rate?",
        "Do paid games do better than free games?",
        "How much does developer track record matter?",
        "What features does the model rely on most?",
        "How did the stacking ensemble perform?",
        "Does supporting more platforms help success?",
    ]

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for i, sample in enumerate(samples):
        col = sample_cols[i % 2]
        if col.button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = sample

    # Display chat history
    st.markdown("---")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle pending question from button click
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask a question about the data...") or pending

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Querying data and asking Groq..."):
                intents = analyze_question_intent(user_input)
                data_context = query_data(df, intents)
                try:
                    answer = ask_llm(client, user_input, data_context)
                except Exception as exc:
                    answer = (
                        f"Sorry, the Groq API call failed: {exc}\n\n"
                        f"Here's the raw data I found for your question:\n\n{data_context}"
                    )
                st.markdown(answer)

                with st.expander("See raw data context passed to the LLM"):
                    st.text(data_context)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()