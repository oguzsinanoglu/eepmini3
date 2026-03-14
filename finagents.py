"""
finagents.py — Standalone module consolidating all agent code from mp3_assignment.ipynb.

Exposes:
    MODEL_SMALL, MODEL_LARGE
    run_single_agent_chat(question, model, history) -> str
    run_multi_agent_chat(question, model, history)  -> str
"""

import os, json, time, sqlite3, requests
import concurrent.futures
import threading as _threading
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────
load_dotenv(override=True)

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY",       "YOUR_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_KEY")

MODEL_SMALL  = "gpt-4o-mini"
MODEL_LARGE  = "gpt-4o"
ACTIVE_MODEL = MODEL_SMALL   # mutable global; set via set_active_model()

client = OpenAI(api_key=OPENAI_API_KEY)

# ── AlphaVantage URL ─────────────────────────────────────────
ALPHAVANTAGE_URL      = "https://www.alphavantage.co"
MOCK_ALPHAVANTAGE_URL = "http://127.0.0.1:2345"
AVURL = MOCK_ALPHAVANTAGE_URL if os.getenv("USE_MOCK_AV_API") == "1" else ALPHAVANTAGE_URL

# ── Database path — resolve relative to this file ────────────
DB_PATH = str(Path(__file__).parent / "stocks.db")

# Tickers that failed a yfinance download this session
_DELISTED_CACHE: set = set()


# ═══════════════════════════════════════════════════════════════
# Tool Functions
# ═══════════════════════════════════════════════════════════════

def get_price_performance(tickers: list, period: str = "1y") -> dict:
    """% price change for a list of tickers over a period."""
    results = {}
    to_download = [t for t in tickers if t not in _DELISTED_CACHE]
    for t in tickers:
        if t in _DELISTED_CACHE:
            results[t] = {"error": "No data — possibly delisted (cached)"}

    if not to_download:
        return results

    _box = [None, None]
    def _run():
        try:
            _box[0] = yf.download(
                to_download, period=period, progress=False,
                auto_adjust=True, group_by="ticker"
            )
        except Exception as _e:
            _box[1] = str(_e)

    _t = _threading.Thread(target=_run, daemon=True)
    _t.start()
    _t.join(timeout=30)

    if _t.is_alive() or _box[1]:
        msg = "Download timed out" if _t.is_alive() else _box[1]
        for t in to_download:
            _DELISTED_CACHE.add(t)
            results[t] = {"error": msg}
        return results

    data = _box[0]
    for ticker in to_download:
        try:
            td = data if len(to_download) == 1 else data[ticker]
            close = td["Close"].dropna()
            if close.empty:
                _DELISTED_CACHE.add(ticker)
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(close.iloc[0])
            end   = float(close.iloc[-1])
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price"  : round(end,   2),
                "pct_change" : round((end - start) / start * 100, 2),
                "period"     : period,
            }
        except Exception as e:
            _DELISTED_CACHE.add(ticker)
            results[ticker] = {"error": str(e)}
    return results


def get_market_status() -> dict:
    """Open/closed status for global stock exchanges."""
    return requests.get(
        f"{AVURL}/query?function=MARKET_STATUS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()


def get_top_gainers_losers() -> dict:
    """Today's top gaining, top losing, and most active tickers."""
    return requests.get(
        f"{AVURL}/query?function=TOP_GAINERS_LOSERS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()


def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    """Latest headlines + Bullish/Bearish/Neutral sentiment for a ticker."""
    data = requests.get(
        f"{AVURL}/query?function=NEWS_SENTIMENT"
        f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title"    : a.get("title"),
                "source"   : a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score"    : a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }


def query_local_db(sql: str) -> dict:
    """Run any SQL SELECT on stocks.db."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


def get_company_overview(ticker: str) -> dict:
    """Fundamentals for one stock: P/E, EPS, market cap, 52-week range."""
    try:
        url  = (f"{AVURL}/query"
                f"?function=OVERVIEW&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}")
        data = requests.get(url, timeout=10).json()
        if not data or "Name" not in data:
            return {"error": f"No overview data found for ticker '{ticker}'"}
        return {
            "ticker"      : ticker,
            "name"        : data.get("Name", ""),
            "sector"      : data.get("Sector", ""),
            "pe_ratio"    : data.get("PERatio", "N/A"),
            "eps"         : data.get("EPS", "N/A"),
            "market_cap"  : data.get("MarketCapitalization", "N/A"),
            "week_high_52": data.get("52WeekHigh", "N/A"),
            "week_low_52" : data.get("52WeekLow", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_tickers_by_sector(sector: str) -> dict:
    """Return stocks in a sector/industry from the local DB."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ticker, company, sector, industry, market_cap "
        "FROM stocks WHERE LOWER(sector) = LOWER(?)",
        conn, params=[sector])
    if df.empty:
        df = pd.read_sql_query(
            "SELECT ticker, company, sector, industry, market_cap "
            "FROM stocks WHERE LOWER(?) LIKE ('%' || LOWER(sector) || '%') "
            "   OR LOWER(sector) LIKE ('%' || LOWER(?) || '%')",
            conn, params=[sector, sector])
    if df.empty:
        df = pd.read_sql_query(
            "SELECT ticker, company, sector, industry, market_cap "
            "FROM stocks WHERE LOWER(industry) LIKE ('%' || LOWER(?) || '%')",
            conn, params=[sector])
    conn.close()
    return {"stocks": df.to_dict(orient="records")}


# ── Dispatch map ──────────────────────────────────────────────
ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector" : get_tickers_by_sector,
    "get_price_performance" : get_price_performance,
    "get_company_overview"  : get_company_overview,
    "get_market_status"     : get_market_status,
    "get_top_gainers_losers": get_top_gainers_losers,
    "get_news_sentiment"    : get_news_sentiment,
    "query_local_db"        : query_local_db,
}


# ═══════════════════════════════════════════════════════════════
# Agent Infrastructure
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    agent_name   : str
    answer       : str
    tools_called : list  = field(default_factory=list)
    raw_data     : dict  = field(default_factory=dict)
    confidence   : float = 0.0
    issues_found : list  = field(default_factory=list)

    def summary(self):
        print(f"[{self.agent_name}] tools={self.tools_called}")
        print(f"  answer: {self.answer[:200]}")


def run_specialist_agent(
    agent_name   : str,
    system_prompt: str,
    task         : str,
    tool_schemas : list,
    max_iters    : int  = 8,
    verbose      : bool = False,
) -> AgentResult:
    """Core agentic loop: system prompt + task → tool calls → final answer."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": task},
    ]
    tools_called = []
    raw_data     = {}

    for i in range(max_iters):
        kwargs = {"model": ACTIVE_MODEL, "messages": messages}
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                if verbose:
                    print(f"  [{agent_name}] iter {i+1}: {fn_name}({fn_args})")
                func   = ALL_TOOL_FUNCTIONS.get(fn_name)
                result = func(**fn_args) if func else {"error": f"Unknown tool: {fn_name}"}
                tools_called.append(fn_name)
                raw_data[f"{fn_name}_{len(tools_called)}"] = result
                messages.append({
                    "role"        : "tool",
                    "tool_call_id": tc.id,
                    "content"     : json.dumps(result, default=str),
                })
        else:
            answer = msg.content or ""
            return AgentResult(
                agent_name=agent_name,
                answer=answer,
                tools_called=tools_called,
                raw_data=raw_data,
            )

    last = messages[-1]
    last_content = (last.get("content", "") if isinstance(last, dict) else (last.content or ""))
    return AgentResult(
        agent_name=agent_name,
        answer=last_content if last_content else "Max iterations reached without final answer.",
        tools_called=tools_called,
        raw_data=raw_data,
    )


# ═══════════════════════════════════════════════════════════════
# Tool Schemas
# ═══════════════════════════════════════════════════════════════

def _s(name, desc, props, req):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req}}}

SCHEMA_TICKERS  = _s("get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Information Technology', 'Energy') or sub-sectors ('semiconductor', 'insurance').",
    {"sector": {"type": "string", "description": "Sector or industry name"}}, ["sector"])

SCHEMA_PRICE    = _s("get_price_performance",
    "Get % price change for a list of tickers over a time period. Periods: '1mo','3mo','6mo','ytd','1y'.",
    {"tickers": {"type": "array", "items": {"type": "string"}},
     "period":  {"type": "string", "default": "1y"}}, ["tickers"])

SCHEMA_OVERVIEW = _s("get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker": {"type": "string", "description": "Ticker symbol e.g. 'AAPL'"}}, ["ticker"])

SCHEMA_STATUS   = _s("get_market_status",
    "Check whether global stock exchanges are currently open or closed.", {}, [])

SCHEMA_MOVERS   = _s("get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.", {}, [])

SCHEMA_NEWS     = _s("get_news_sentiment",
    "Get latest news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {"ticker": {"type": "string"}, "limit": {"type": "integer", "default": 5}}, ["ticker"])

SCHEMA_SQL      = _s("query_local_db",
    "Run a SQL SELECT on stocks.db. Table 'stocks': ticker, company, sector, industry, market_cap (Large/Mid/Small), exchange.",
    {"sql": {"type": "string", "description": "A valid SQL SELECT statement"}}, ["sql"])

ALL_SCHEMAS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW,
               SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL]


# ═══════════════════════════════════════════════════════════════
# Baseline Agent
# ═══════════════════════════════════════════════════════════════

def run_baseline(question: str, verbose: bool = False) -> AgentResult:
    return run_specialist_agent(
        agent_name="Baseline",
        system_prompt=(
            "You are a financial analyst assistant. Answer the user's question "
            "to the best of your knowledge. You have no access to external tools, "
            "live data, or databases. If you are unsure about specific current values, "
            "say so clearly."
        ),
        task=question,
        tool_schemas=[],
        max_iters=1,
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════
# Single Agent
# ═══════════════════════════════════════════════════════════════

SINGLE_AGENT_PROMPT = """You are an expert financial analyst assistant with access to 7 tools for retrieving real-time financial data.

YOUR JOB: Answer user questions accurately using data retrieved from your tools. Trust the data your tools return — it comes from live APIs and databases.

AVAILABLE TOOLS AND WHEN TO USE THEM:
1. get_tickers_by_sector — Look up companies by sector or industry (e.g., "tech stocks", "semiconductors", "energy companies"). ALWAYS use this first when a question mentions a sector/industry rather than specific tickers.
2. get_price_performance — Get % price change for one or more tickers over a period (1mo, 3mo, 6mo, ytd, 1y).
3. get_company_overview — Get fundamentals: P/E ratio, EPS, market cap, 52-week high/low for a single ticker.
4. get_market_status — Check if US stock markets are currently open or closed.
5. get_top_gainers_losers — Get today's top movers (gainers, losers, most active).
6. get_news_sentiment — Get recent news headlines and sentiment scores for a ticker.
7. query_local_db — Run SQL on the local stocks database (columns: ticker, company, sector, industry, market_cap, exchange). Use for filtering by market_cap='Large', exchange='NASDAQ', etc.

CRITICAL RULES:
- For sector/industry questions: FIRST look up tickers with get_tickers_by_sector or query_local_db, THEN fetch data for those tickers. Never guess tickers.
- For comparison questions: Fetch data for ALL tickers mentioned, then compare.
- For multi-condition questions (e.g., "dropped this month but grew this year"): Fetch data for BOTH time periods, then filter and compare.
- When a tool returns data, TRUST it and report it directly.
- If a tool returns an error or empty data, say so explicitly. Do not guess values when a tool fails.
- Present numerical results clearly with the data source identified.
- When ranking or filtering results, show the actual numbers you used to make the determination.
- Use the exact numerical values provided by the tools. DO NOT round or APPROXIMATE.

CONVERSATIONAL CONTEXT: If the user refers to "that", "it", "the two", or similar pronouns, use the conversation history provided to resolve the reference before fetching data."""


def run_single_agent(question: str, verbose: bool = False) -> AgentResult:
    return run_specialist_agent(
        agent_name="Single Agent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=question,
        tool_schemas=ALL_SCHEMAS,
        max_iters=10,
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════
# Multi-Agent System
# ═══════════════════════════════════════════════════════════════

MARKET_TOOLS      = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_STATUS, SCHEMA_MOVERS]
FUNDAMENTAL_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_SQL, SCHEMA_TICKERS]
SENTIMENT_TOOLS   = [SCHEMA_NEWS, SCHEMA_SQL]

ORCHESTRATOR_PROMPT = """You are a query router for a financial multi-agent system.
Given a user question, you decide which specialist agents to activate and write a focused sub-task for each one.

Specialists available:
- "Price"        — handles price performance, market status, top gainers/losers, sector tickers
- "Fundamentals" — handles P/E ratio, EPS, market cap, 52-week range, DB sector queries
- "Sentiment"    — handles news headlines and Bullish/Bearish/Neutral sentiment scores

RULES:
1. Only activate specialists that are strictly needed to answer the question.
2. Detect a cross-domain ranking dependency: if the question first ranks by price/return and then asks for fundamentals or sentiment of the top results (a DIFFERENT domain), set "phased": true and name the Phase 1 agent as "phase1_agent": "Price".
3. Write a concise, self-contained sub-task string for each activated specialist. The sub-task must include any ticker symbols mentioned in the original question.
4. Return ONLY valid JSON — no prose before or after.
5. Always activate at least one specialist — never return an empty "agents" list. If the question is purely about DB/sector lookup (e.g., "list companies in database"), activate "Price" (it has get_tickers_by_sector).
6. For questions requiring TWO time-period conditions on the SAME stocks (e.g., "dropped this month but grew this year"), set "phased": false, assign ONLY "Price", and write the sub-task as: "Fetch 1mo AND ytd performance for [sector] tickers. Filter: 1mo<0 AND ytd>0. Return top 3 by ytd."

Return format:
{
  "agents": ["Price", "Fundamentals"],
  "phased": false,
  "phase1_agent": "",
  "task_per_agent": {
    "Price":        "...",
    "Fundamentals": "..."
  }
}"""

PRICE_AGENT_PROMPT = """You are a market-data specialist with access to price performance, market status, top movers, and sector/ticker lookup tools.
Answer accurately using only the data your tools return. If a tool fails, say so.
Do not guess ticker symbols — use get_tickers_by_sector to look them up first when needed.

RANKING PROTOCOL — follow these steps exactly when asked to rank stocks by return:
1. Call get_price_performance ONCE with ALL relevant tickers in a single call.
2. Build an internal scratchpad: list EVERY returned ticker and its exact pct_change value.
3. Sort the scratchpad from highest to lowest pct_change in your reasoning.
4. Select the top-N entries from the sorted list.
5. Write your final answer using the EXACT ticker symbol paired with its EXACT pct_change value.

STRICT TICKER RULES:
- The KEY in the get_price_performance response IS the ticker. Its value belongs only to that ticker.
- Never assign one ticker's pct_change to a different ticker symbol.
- Never infer or guess a return value from your training knowledge.

OUTPUT FORMAT:
A) COMPARISON (multiple specific tickers): Report ALL requested tickers: TICKER: start=$X.XX  end=$X.XX  change=+Y.YY%
B) SINGLE TICKER: TICKER: start=$X.XX  end=$X.XX  change=+Y.YY%
C) SECTOR RANKING (top-N): List top-N from sorted scratchpad: 1. TICKER: +X.XX%

MULTI-CONDITION FILTERING PROTOCOL:
1. Call get_tickers_by_sector to get the ticker list if not provided.
2. Call get_price_performance ONCE for period=1mo with ALL tickers.
3. Call get_price_performance ONCE for period=ytd with ALL tickers.
4. Build a dual scratchpad. Filter and sort. Report both 1mo% and ytd%.

Always report the exact numeric pct_change for every stock you mention."""

FUNDAMENTALS_AGENT_PROMPT = """You are a fundamental analysis specialist with access to company overview data (P/E, EPS, market cap, 52-week range) and a local stocks database.
Answer accurately using only the data your tools return. If a tool fails, say so.
Present all values clearly with the field name (e.g. "P/E ratio: 28.5").
IMPORTANT: Only fetch data for the specific tickers listed in your task. Do not expand to other tickers not mentioned."""

SENTIMENT_AGENT_PROMPT = """You are a news and sentiment specialist with access to real-time news headlines and sentiment scores for individual stocks.
Summarise sentiment clearly: Bullish / Bearish / Neutral with score.
If a tool fails or returns no articles, say so explicitly. Do not fabricate headlines.
IMPORTANT: Only fetch sentiment for the specific tickers listed in your task."""

CRITIC_PROMPT = """You are a financial fact-checker evaluating a single specialist agent.
You are given the agent's answer and the raw tool data it actually retrieved.
Your ONLY job: check whether the answer is internally consistent with the raw_data_sample.

STRICT RULES:
- You see data for ONE agent only. Evaluate ONLY that agent's answer against its own raw data.
- Do NOT compare with other agents, other tickers, or your training knowledge.
- Do NOT flag tickers the agent was not asked about.
- Do NOT flag a ticker as "missing data" if that ticker failed a download — this is expected.
- If the raw data says a value, treat it as ground truth.
- Only flag an issue if the agent's answer contradicts or omits something in its own raw_data_sample.
- Do NOT flag minor rounding differences (< 5% relative error).

Output a single JSON object:
{
  "agent_name": "...",
  "confidence": 0.85,
  "issues_found": ["...", "..."]
}
IMPORTANT: Return ONLY the raw JSON object. No markdown, no code fences, no prose."""

SYNTHESIZER_PROMPT = """You are the final answer synthesizer for a financial multi-agent system.
You receive a JSON object with three fields:
  original_question   — the user's question
  price_returns       — dict of {TICKER: pct_change_float}; may be empty
  specialist_answers  — list of {agent, answer, confidence, issues}

OUTPUT RULES:

1. TABLE FORMAT — use ONLY when price_returns contains 2 or more tickers:
   - Output exactly one line per ticker. FORMAT: TICKER: +X.XX% | P/E X.XX | Sentiment: Label (score)
   - Use the exact float from price_returns for %. Prefix positive with +.
   - Get P/E and Sentiment from specialist_answers. Write N/A only if truly absent.
   - No company names, no intro sentence, no conclusion, no blank lines.

2. PROSE FORMAT — use for all other questions:
   - Include ALL key numeric values from the specialist answers.
   - For SENTIMENT: list EVERY headline with its exact label and score.
   - For SINGLE-TICKER PRICE: report start, end, AND % change.
   - For MULTI-CONDITION FILTER: list every qualifying stock with BOTH 1-month % AND YTD %.
   - Draw facts from specialist_answers only.
   - No markdown, no bullet points, no headers.
   - Use numerical values exactly as provided.
   - If a required data field is missing from all specialist answers, say so inline briefly."""


def run_orchestrator(question: str) -> dict:
    """One LLM call to plan which specialists to activate."""
    response = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user",   "content": question},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    plan = json.loads(raw)
    # Normalise
    if not plan.get("agents"):
        plan["agents"] = ["Price", "Fundamentals", "Sentiment"]
    if "phased" not in plan:
        plan["phased"] = False
    if "phase1_agent" not in plan:
        plan["phase1_agent"] = ""
    if "task_per_agent" not in plan:
        plan["task_per_agent"] = {a: question for a in plan["agents"]}
    return plan


def _run_price_agent(task: str, verbose: bool = False) -> AgentResult:
    return run_specialist_agent("Price Agent", PRICE_AGENT_PROMPT, task, MARKET_TOOLS, verbose=verbose)

def _run_fundamentals_agent(task: str, verbose: bool = False) -> AgentResult:
    return run_specialist_agent("Fundamentals Agent", FUNDAMENTALS_AGENT_PROMPT, task, FUNDAMENTAL_TOOLS, verbose=verbose)

def _run_sentiment_agent(task: str, verbose: bool = False) -> AgentResult:
    return run_specialist_agent("Sentiment Agent", SENTIMENT_AGENT_PROMPT, task, SENTIMENT_TOOLS, verbose=verbose)

SPECIALIST_RUNNERS = {
    "Price"       : _run_price_agent,
    "Fundamentals": _run_fundamentals_agent,
    "Sentiment"   : _run_sentiment_agent,
}


def _compress_for_critic(val) -> str:
    text = json.dumps(val, default=str)
    return text[:3000] if len(text) > 3000 else text


def _critic_one(result: AgentResult, verbose: bool) -> None:
    payload = {
        "agent_answer": result.answer[:1500],
        "raw_data_sample": {k: _compress_for_critic(v) for k, v in list(result.raw_data.items())[:5]},
    }
    response = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user",   "content": json.dumps(payload)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        critique = json.loads(raw)
        result.confidence   = float(critique.get("confidence", 0.7))
        result.issues_found = critique.get("issues_found", [])
    except Exception:
        result.confidence = 0.7
        result.issues_found = []


def run_critic(question: str, agent_results: list, verbose: bool = False) -> None:
    for r in agent_results:
        _critic_one(r, verbose)


def run_synthesizer(question: str, agent_results: list, price_returns: dict = None) -> str:
    payload = {
        "original_question" : question,
        "price_returns"     : price_returns or {},
        "specialist_answers": [
            {"agent": r.agent_name, "answer": r.answer,
             "confidence": r.confidence, "issues": r.issues_found}
            for r in agent_results
        ],
    }
    response = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {"role": "user",   "content": json.dumps(payload)},
        ],
    )
    return response.choices[0].message.content or ""


def _extract_tickers_from_result(result: AgentResult, top_n: int = 3) -> list:
    tickers = []
    for tool_output in result.raw_data.values():
        if not isinstance(tool_output, dict):
            continue
        for k, v in tool_output.items():
            if (isinstance(k, str) and k.isupper() and 1 <= len(k) <= 5
                    and isinstance(v, dict) and "pct_change" in v):
                tickers.append((k, float(v["pct_change"])))
    tickers.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in tickers[:top_n]]


def run_multi_agent(question: str, verbose: bool = False) -> dict:
    """Orchestrator → Parallel Specialists → Critic → Synthesizer pipeline."""
    t0 = time.time()
    plan = run_orchestrator(question)

    agent_results: list[AgentResult] = []
    price_returns: dict = {}

    if plan.get("phased") and plan.get("phase1_agent"):
        p1_name = plan["phase1_agent"]
        p1_task = plan["task_per_agent"].get(p1_name, question)
        p1_result = SPECIALIST_RUNNERS[p1_name](p1_task, verbose=verbose)
        agent_results.append(p1_result)

        top_tickers = _extract_tickers_from_result(p1_result, top_n=3)
        _all_returns = {
            k: round(float(v["pct_change"]), 2)
            for to in p1_result.raw_data.values() if isinstance(to, dict)
            for k, v in to.items()
            if isinstance(k, str) and k.isupper() and 1 <= len(k) <= 5
            and isinstance(v, dict) and "pct_change" in v
        }
        price_returns = {t: _all_returns[t] for t in top_tickers if t in _all_returns}
        ticker_hint = (f" Use ONLY these tickers: {', '.join(top_tickers)}." if top_tickers else "")

        phase2_agents = [a for a in plan["agents"] if a != p1_name]
        if phase2_agents:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(phase2_agents))) as ex:
                futures = {
                    ex.submit(
                        SPECIALIST_RUNNERS[name],
                        plan["task_per_agent"].get(name, question) + ticker_hint,
                        verbose,
                    ): name for name in phase2_agents
                }
                for future in concurrent.futures.as_completed(futures):
                    agent_results.append(future.result())
    else:
        active_agents = plan.get("agents") or ["Price", "Fundamentals", "Sentiment"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(active_agents))) as ex:
            futures = {
                ex.submit(
                    SPECIALIST_RUNNERS[name],
                    plan["task_per_agent"].get(name, question),
                    verbose,
                ): name for name in active_agents
            }
            for future in concurrent.futures.as_completed(futures):
                agent_results.append(future.result())

        for r in agent_results:
            if r.agent_name == "Price Agent":
                for tool_output in r.raw_data.values():
                    if isinstance(tool_output, dict):
                        for k, v in tool_output.items():
                            if (isinstance(k, str) and k.isupper()
                                    and 1 <= len(k) <= 5
                                    and isinstance(v, dict) and "pct_change" in v):
                                price_returns[k] = round(float(v["pct_change"]), 2)

    run_critic(question, agent_results, verbose=verbose)
    final_answer = run_synthesizer(question, agent_results, price_returns=price_returns)

    return {
        "final_answer" : final_answer,
        "agent_results": agent_results,
        "elapsed_sec"  : time.time() - t0,
        "architecture" : "orchestrator-critic",
    }


# ═══════════════════════════════════════════════════════════════
# Public Chat API — used by app.py
# ═══════════════════════════════════════════════════════════════

def set_active_model(model: str) -> None:
    """Set the global ACTIVE_MODEL used by all agent functions."""
    global ACTIVE_MODEL
    ACTIVE_MODEL = model


def _build_contextual_question(question: str, history: list) -> str:
    """
    Prepend the last few conversation turns to the current question so agents
    can resolve pronouns and follow-up references.

    history items: {"role": "user"|"assistant", "content": str}
    """
    if not history:
        return question
    # Include up to the last 6 messages (3 exchanges)
    recent = history[-6:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    context = "\n".join(lines)
    return (
        f"Conversation history (for context only — do NOT re-answer old questions):\n"
        f"{context}\n\n"
        f"New question: {question}"
    )


def run_single_agent_chat(question: str, model: str, history: list = None) -> str:
    """
    Run the single agent with conversational context.
    Returns the final answer string.
    """
    set_active_model(model)
    full_q = _build_contextual_question(question, history or [])
    result = run_single_agent(full_q, verbose=False)
    return result.answer


def run_multi_agent_chat(question: str, model: str, history: list = None) -> str:
    """
    Run the multi-agent pipeline with conversational context.
    Returns the final answer string.
    """
    set_active_model(model)
    full_q = _build_contextual_question(question, history or [])
    result = run_multi_agent(full_q, verbose=False)
    return result["final_answer"]
