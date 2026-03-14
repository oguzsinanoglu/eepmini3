"""
app.py — Streamlit chat interface for Mini Project 3 financial agents.

Run with:  streamlit run app.py
"""

import os
import threading
import time
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

# ── Start mock AV server once per worker process ─────────────
@st.cache_resource
def _start_mock_av_server():
    if os.environ.get("USE_MOCK_AV_API") != "1":
        return False
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    from av_mock_server import app as _flask_app
    t = threading.Thread(
        target=lambda: _flask_app.run(host="127.0.0.1", port=2345, use_reloader=False),
        daemon=True,
    )
    t.start()
    time.sleep(0.8)   # let Flask bind before finagents starts sending requests
    return True

_start_mock_av_server()

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
