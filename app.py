"""
app.py — Streamlit chat interface for Mini Project 3 financial agents.

Run with:  streamlit run app.py
"""

import os
import json
import streamlit as st

# ── Page config — must be first st.* call ─────────────────────
st.set_page_config(
    page_title="FinAgent Chat",
    page_icon="📈",
    layout="wide",
)

# ── Bridge st.secrets → os.environ so finagents.py's os.getenv() works ──────
for _k in ["OPENAI_API_KEY", "ALPHAVANTAGE_API_KEY", "USE_MOCK_AV_API"]:
    if _k in st.secrets and not os.environ.get(_k):
        os.environ[_k] = str(st.secrets[_k])

# ── Patch requests to intercept mock AV calls (no Flask thread needed) ───────
@st.cache_resource
def _patch_mock_av_requests():
    """
    Monkey-patch requests.get to intercept calls to http://127.0.0.1:2345
    and route them directly to av_mock_server handler functions.
    Avoids all Flask threading and port-binding issues on Streamlit Cloud.

    Note: finagents.py calls requests.get() (the module-level function), not
    Session.get(), so we must patch the module-level function.
    """
    if os.environ.get("USE_MOCK_AV_API") != "1":
        return False
    import requests as _req
    from urllib.parse import urlparse, parse_qs
    from av_mock_server import (
        _handle_overview,
        _handle_market_status,
        _handle_top_gainers_losers,
        _handle_news_sentiment,
    )

    _orig_get = _req.get

    _HANDLERS = {
        "OVERVIEW":           _handle_overview,
        "MARKET_STATUS":      _handle_market_status,
        "TOP_GAINERS_LOSERS": _handle_top_gainers_losers,
        "NEWS_SENTIMENT":     _handle_news_sentiment,
    }

    def _intercepted_get(url, **kwargs):
        parsed = urlparse(url)
        if parsed.hostname == "127.0.0.1" and parsed.port == 2345:
            params = {k: v[0] for k, v in parse_qs(parsed.query, keep_blank_values=True).items()}
            fn = params.get("function", "")
            handler = _HANDLERS.get(fn, lambda p: {"error": f"Unknown function: {fn}"})
            data = handler(params)
            resp = _req.models.Response()
            resp.status_code = 200
            resp._content = json.dumps(data).encode("utf-8")
            resp.encoding = "utf-8"
            return resp
        return _orig_get(url, **kwargs)

    _req.get = _intercepted_get
    return True

_patch_mock_av_requests()

# ── Import agents after env vars are set ─────────────────────
from finagents import run_single_agent_chat, run_multi_agent_chat, MODEL_SMALL, MODEL_LARGE  # noqa: E402


# ── Session state initialisation ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role", "content", "metadata"}


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    agent_choice = st.selectbox(
        "Agent Architecture",
        options=["Single Agent", "Multi-Agent"],
        index=0,
    )

    model_choice = st.selectbox(
        "Model",
        options=[MODEL_SMALL, MODEL_LARGE],
        index=0,
    )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "**How it works**\n\n"
        "Single Agent — one LLM with all 7 tools.\n\n"
        "Multi-Agent — Orchestrator routes to Price / Fundamentals / "
        "Sentiment specialists, a Critic verifies each answer, "
        "then a Synthesizer merges them."
    )


# ── Main area ─────────────────────────────────────────────────
st.title("📈 FinAgent Chat")
st.caption("Ask questions about stock prices, fundamentals, and market sentiment.")

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            st.caption(
                f"🤖 {meta['agent']} · 🧠 {meta['model']}"
            )


# ── Input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about stocks, P/E ratios, sentiment…"):

    # Store and show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build history list (exclude metadata — agents only need role+content)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]   # exclude the message just added
    ]

    # Call the selected agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if agent_choice == "Single Agent":
                    answer = run_single_agent_chat(prompt, model_choice, history)
                else:
                    answer = run_multi_agent_chat(prompt, model_choice, history)
            except Exception as e:
                answer = f"❌ Error: {e}"

        st.markdown(answer)
        st.caption(f"🤖 {agent_choice} · 🧠 {model_choice}")

    # Persist assistant message
    st.session_state.messages.append({
        "role"    : "assistant",
        "content" : answer,
        "metadata": {"agent": agent_choice, "model": model_choice},
    })
