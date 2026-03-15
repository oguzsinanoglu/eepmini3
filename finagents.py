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

# Merge with av_mock_server's delisted set if already loaded (they share process memory)
def _get_delisted_cache() -> set:
    """Return the union of the local cache and av_mock_server's cache."""
    try:
        import av_mock_server as _avms
        return _DELISTED_CACHE | _avms._delisted_tickers
    except Exception:
        return _DELISTED_CACHE


# ═══════════════════════════════════════════════════════════════
# Tool Functions
# ═══════════════════════════════════════════════════════════════

def get_price_performance(tickers: list, period: str = "1y") -> dict:
    """% price change for a list of tickers over a period."""
    results = {}
    to_download = [t for t in tickers if t not in _get_delisted_cache()]
    for t in tickers:
        if t in _get_delisted_cache():
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
            "ticker"        : ticker,
            "name"          : data.get("Name", ""),
            "sector"        : data.get("Sector", ""),
            "pe_ratio"      : data.get("PERatio", "N/A"),
            "eps"           : data.get("EPS", "N/A"),
            "market_cap"    : data.get("MarketCapitalization", "N/A"),
            "week_high_52"  : data.get("52WeekHigh", "N/A"),
            "week_low_52"   : data.get("52WeekLow", "N/A"),
            "current_price" : data.get("CurrentPrice", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_tickers_by_sector(sector: str) -> dict:
    """Return stocks in a sector/industry from the local DB."""
    sector = _normalize_sector(sector)
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


# Map common colloquial sector names to the exact strings used in sp500_companies.csv
_SECTOR_ALIASES = {
    "finance"      : "Financial Services",
    "financial"    : "Financial Services",
    "financials"   : "Financial Services",
    "tech"         : "Technology",
    "it"           : "Technology",
    "information technology": "Technology",
    "comms"        : "Communication Services",
    "communications": "Communication Services",
    "telecom"      : "Communication Services",
    "consumer"     : "Consumer Cyclical",
    "health"       : "Healthcare",
    "health care"  : "Healthcare",
    "material"     : "Basic Materials",
    "materials"    : "Basic Materials",
    "industrial"   : "Industrials",
    "real estate"  : "Real Estate",
    "reits"        : "Real Estate",
    "utility"      : "Utilities",
}

def _normalize_sector(sector: str) -> str:
    """Map colloquial sector names to the canonical strings in stocks.db."""
    return _SECTOR_ALIASES.get(sector.strip().lower(), sector)


def rank_stocks_by_metric(
    sector: str,
    metric: str = "pe_ratio",
    top_n: int = 5,
    descending: bool = True,
    market_cap: str = "Large",
) -> dict:
    """Fetch Large-cap tickers for a sector, get their overview, sort by metric,
    and return the top-N ranked list. Sorting is done in Python — not by the LLM.

    metric: one of 'pe_ratio', 'eps', 'market_cap' (MarketCapitalization), 'week_high_52'.
    """
    sector = _normalize_sector(sector)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ticker FROM stocks "
        "WHERE LOWER(sector) LIKE ('%' || LOWER(?) || '%') "
        "  AND market_cap = ? ORDER BY ticker LIMIT 20",
        conn, params=[sector, market_cap])
    conn.close()

    if df.empty:
        return {"error": f"No {market_cap}-cap stocks found for sector '{sector}'"}

    tickers = df["ticker"].tolist()

    # 2. Fetch overview for each ticker (skip delisted)
    rows = []
    for t in tickers[:15]:
        ov = get_company_overview(t)
        if "error" in ov:
            continue
        # Parse the requested metric to float for sorting
        raw = ov.get(metric, "None")
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue  # metric is None / N/A — skip
        rows.append({**ov, "_sort_val": val})

    if not rows:
        return {"error": f"No valid '{metric}' data found for {market_cap}-cap stocks in '{sector}'"}

    # 3. Sort in Python — deterministic, no LLM involvement
    rows.sort(key=lambda r: r["_sort_val"], reverse=descending)
    ranked = []
    for rank, r in enumerate(rows[:top_n], start=1):
        ranked.append({
            "rank"        : rank,
            "ticker"      : r["ticker"],
            "name"        : r.get("name", ""),
            "pe_ratio"    : r.get("pe_ratio", "N/A"),
            "eps"         : r.get("eps", "N/A"),
            "market_cap"  : r.get("market_cap", "N/A"),
            "week_high_52": r.get("week_high_52", "N/A"),
            "week_low_52" : r.get("week_low_52", "N/A"),
        })

    return {"ranked": ranked, "metric": metric, "sector": sector, "descending": descending}


def filter_sector_by_52week(sector: str, market_cap: str = "Large", max_results: int = 8) -> dict:
    """Fetch stocks for a sector, get their overview, and return those trading
    closer to their 52-week low than their 52-week high (current_price < midpoint).
    Returns up to max_results stocks sorted by pct_above_low ascending (closest to low first).
    Filtering is done in Python — no LLM guessing.
    """
    sector = _normalize_sector(sector)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ticker FROM stocks "
        "WHERE LOWER(sector) LIKE ('%' || LOWER(?) || '%') "
        "  AND market_cap = ? ORDER BY ticker LIMIT 30",
        conn, params=[sector, market_cap])
    conn.close()

    if df.empty:
        return {"error": f"No {market_cap}-cap stocks found for sector '{sector}'"}

    qualifying = []
    for t in df["ticker"].tolist():
        ov = get_company_overview(t)
        if "error" in ov:
            continue
        try:
            high = float(ov.get("week_high_52", "None"))
            low  = float(ov.get("week_low_52",  "None"))
            cur  = float(ov.get("current_price", "None"))
        except (TypeError, ValueError):
            continue
        if cur < (high + low) / 2:  # closer to 52-week low
            qualifying.append({
                "ticker"       : t,
                "name"         : ov.get("name", ""),
                "current_price": cur,
                "week_high_52" : high,
                "week_low_52"  : low,
                "pct_above_low": round((cur - low) / low * 100, 2) if low else None,
            })

    qualifying.sort(key=lambda r: r["pct_above_low"] if r["pct_above_low"] is not None else 999)
    if not qualifying:
        return {"result": "No stocks found trading closer to their 52-week low.", "sector": sector}

    return {"qualifying": qualifying[:max_results], "sector": sector, "count": len(qualifying)}


def filter_sector_by_return_conditions(
    sector: str,
    period1: str = "1mo",
    condition1: str = "negative",
    period2: str = "ytd",
    condition2: str = "positive",
    sort_by: str = "period2",
    top_n: int = 3,
    market_cap: str = "Large",
) -> dict:
    """Fetch ALL tickers for a sector, download period1 and period2 returns for
    every ticker, filter in Python by condition1/condition2, sort, and return top_n.
    Filtering is deterministic — no LLM arithmetic.

    condition values: 'positive' (>0) or 'negative' (<0)
    sort_by values: 'period1' or 'period2'
    """
    sector = _normalize_sector(sector)
    conn = sqlite3.connect(DB_PATH)
    if market_cap:
        df = pd.read_sql_query(
            "SELECT ticker, company FROM stocks "
            "WHERE LOWER(sector) LIKE ('%' || LOWER(?) || '%') "
            "  AND market_cap = ? ORDER BY ticker LIMIT 60",
            conn, params=[sector, market_cap])
    else:
        df = pd.read_sql_query(
            "SELECT ticker, company FROM stocks "
            "WHERE LOWER(sector) LIKE ('%' || LOWER(?) || '%') "
            "ORDER BY ticker LIMIT 60",
            conn, params=[sector])
    conn.close()

    if df.empty:
        return {"error": f"No stocks found for sector '{sector}'"}

    name_map = dict(zip(df["ticker"], df["company"]))
    tickers = df["ticker"].tolist()

    r1 = get_price_performance(tickers, period=period1)
    r2 = get_price_performance(tickers, period=period2)

    def _pct(result, t):
        v = result.get(t, {})
        if isinstance(v, dict) and "pct_change" in v:
            return float(v["pct_change"])
        return None

    def _passes(val, cond):
        if val is None:
            return False
        return val < 0 if cond == "negative" else val > 0

    rows = []
    for t in tickers:
        p1 = _pct(r1, t)
        p2 = _pct(r2, t)
        if _passes(p1, condition1) and _passes(p2, condition2):
            rows.append({"ticker": t, "_p1": p1, "_p2": p2})

    sort_field = "_p1" if sort_by == "period1" else "_p2"
    rows.sort(key=lambda r: r[sort_field], reverse=(condition2 == "positive"))

    formatted = [
        {"ticker": r["ticker"], "name": name_map.get(r["ticker"], ""), period1: f"{r['_p1']:+.2f}", period2: f"{r['_p2']:+.2f}"}
        for r in rows[:top_n]
    ]
    return {
        "results": formatted,
        "total_qualifying": len(rows),
        "sector": sector,
        "filter": f"{period1} {condition1} AND {period2} {condition2}",
    }


def filter_sector_by_min_return(
    sector: str,
    period: str = "ytd",
    min_pct: float = 20.0,
    top_n: int = 5,
    market_cap: str = "Large",
) -> dict:
    """Fetch ALL tickers for a sector, download returns for one period, filter to
    those with pct_change >= min_pct, sort descending, return top_n.
    Filtering is deterministic — no LLM arithmetic.
    """
    sector = _normalize_sector(sector)
    conn = sqlite3.connect(DB_PATH)
    if market_cap:
        df = pd.read_sql_query(
            "SELECT ticker, company FROM stocks "
            "WHERE LOWER(sector) LIKE ('%' || LOWER(?) || '%') "
            "  AND market_cap = ? ORDER BY ticker LIMIT 60",
            conn, params=[sector, market_cap])
    else:
        df = pd.read_sql_query(
            "SELECT ticker, company FROM stocks "
            "WHERE LOWER(sector) LIKE ('%' || LOWER(?) || '%') "
            "ORDER BY ticker LIMIT 60",
            conn, params=[sector])
    conn.close()

    if df.empty:
        return {"error": f"No stocks found for sector '{sector}'"}

    name_map = dict(zip(df["ticker"], df["company"]))
    tickers = df["ticker"].tolist()
    perf = get_price_performance(tickers, period=period)

    rows = []
    for t in tickers:
        v = perf.get(t, {})
        if isinstance(v, dict) and "pct_change" in v:
            pct = float(v["pct_change"])
            if pct >= min_pct:
                rows.append({"ticker": t, "_pct": pct})

    rows.sort(key=lambda r: r["_pct"], reverse=True)
    formatted = [
        {"ticker": r["ticker"], "name": name_map.get(r["ticker"], ""), period: f"{r['_pct']:+.2f}"}
        for r in rows[:top_n]
    ]
    return {
        "results": formatted,
        "total_qualifying": len(rows),
        "sector": sector,
        "filter": f"{period} >= {min_pct}%",
    }


def rank_sector_by_return(
    sector: str,
    period: str = "1y",
    top_n: int = 3,
    market_cap: str = "Large",
) -> dict:
    """Fetch ALL tickers for a sector/industry, download price returns for ONE period,
    sort descending, and return the top-N ranked by return.
    Use for 'top N by 1-year return', 'best performers this year', etc.
    Fully deterministic — no LLM sorting required.
    """
    sector = _normalize_sector(sector)
    conn = sqlite3.connect(DB_PATH)
    if market_cap:
        df = pd.read_sql_query(
            "SELECT ticker, company FROM stocks "
            "WHERE (LOWER(sector) LIKE ('%' || LOWER(?) || '%') "
            "   OR  LOWER(industry) LIKE ('%' || LOWER(?) || '%')) "
            "  AND market_cap = ? ORDER BY ticker LIMIT 60",
            conn, params=[sector, sector, market_cap])
    else:
        df = pd.read_sql_query(
            "SELECT ticker, company FROM stocks "
            "WHERE  LOWER(sector) LIKE ('%' || LOWER(?) || '%') "
            "   OR  LOWER(industry) LIKE ('%' || LOWER(?) || '%') "
            "ORDER BY ticker LIMIT 60",
            conn, params=[sector, sector])
    conn.close()

    if df.empty:
        return {"error": f"No stocks found for sector/industry '{sector}'"}

    name_map = dict(zip(df["ticker"], df["company"]))
    tickers = df["ticker"].tolist()
    perf = get_price_performance(tickers, period=period)

    rows = []
    for t in tickers:
        v = perf.get(t, {})
        if isinstance(v, dict) and "pct_change" in v:
            rows.append({"ticker": t, "_pct": float(v["pct_change"])})

    rows.sort(key=lambda r: r["_pct"], reverse=True)
    return {
        "results": [
            {"ticker": r["ticker"], "name": name_map.get(r["ticker"], ""), period: f"{r['_pct']:+.2f}"}
            for r in rows[:top_n]
        ],
        "total_ranked": len(rows),
        "sector": sector,
        "period": period,
    }


ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector"             : get_tickers_by_sector,
    "get_price_performance"             : get_price_performance,
    "get_company_overview"              : get_company_overview,
    "rank_stocks_by_metric"             : rank_stocks_by_metric,
    "filter_sector_by_52week"           : filter_sector_by_52week,
    "filter_sector_by_return_conditions": filter_sector_by_return_conditions,
    "filter_sector_by_min_return"       : filter_sector_by_min_return,
    "rank_sector_by_return"             : rank_sector_by_return,
    "get_market_status"                 : get_market_status,
    "get_top_gainers_losers"            : get_top_gainers_losers,
    "get_news_sentiment"                : get_news_sentiment,
    "query_local_db"                    : query_local_db,
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
        kwargs = {"model": ACTIVE_MODEL, "messages": messages, "temperature": 0}
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

SCHEMA_RANK     = _s("rank_stocks_by_metric",
    "Fetch Large-cap stocks for a sector, retrieve their fundamentals, sort by a metric in Python, and return the top-N ranked list. "
    "Use this for any 'top N by P/E / EPS / market cap' sector question. Sorting is guaranteed correct.",
    {
        "sector"     : {"type": "string", "description": "Sector name, e.g. 'technology'"},
        "metric"     : {"type": "string", "enum": ["pe_ratio", "eps", "market_cap", "week_high_52"], "description": "Field to rank by"},
        "top_n"      : {"type": "integer", "default": 5, "description": "How many stocks to return"},
        "descending" : {"type": "boolean", "default": True, "description": "True = highest first"},
        "market_cap" : {"type": "string", "enum": ["Large", "Mid", "Small"], "default": "Large"},
    }, ["sector", "metric"])

SCHEMA_52WEEK   = _s("filter_sector_by_52week",
    "Return stocks in a sector that are trading closer to their 52-week low than their 52-week high. "
    "Results are sorted by pct_above_low ascending (closest to 52-week low first). "
    "Use this for '52-week low proximity' questions.",
    {
        "sector"     : {"type": "string", "description": "Sector name, e.g. 'Financial Services'"},
        "market_cap" : {"type": "string", "enum": ["Large", "Mid", "Small"], "default": "Large"},
        "max_results": {"type": "integer", "default": 8, "description": "Max stocks to return (default 8)"},
    }, ["sector"])

SCHEMA_RANK_RETURN = _s("rank_sector_by_return",
    "Fetch ALL tickers for a sector or industry (e.g. 'semiconductor'), download price returns for "
    "ONE period, sort descending, and return the top-N ranked by return. "
    "Use for questions like 'top 3 semiconductor stocks by 1-year return' or "
    "'best performing energy stocks this year'. Works on both broad sectors and sub-industries. "
    "Sorting is guaranteed correct — no LLM arithmetic.",
    {
        "sector"    : {"type": "string", "description": "Sector or industry name, e.g. 'semiconductor', 'technology', 'energy'"},
        "period"    : {"type": "string", "default": "1y", "description": "Period: '1mo','3mo','6mo','ytd','1y'"},
        "top_n"     : {"type": "integer", "default": 3, "description": "How many top stocks to return"},
        "market_cap": {"type": "string", "enum": ["Large", "Mid", "Small", ""], "default": "Large"},
    }, ["sector"])

SCHEMA_MIN_RETURN = _s("filter_sector_by_min_return",
    "Fetch ALL tickers for a sector, download returns for ONE period, filter to those "
    "with pct_change >= min_pct, sort descending, and return top-N. "
    "Use this for questions like 'which stocks grew more than 20% this year' or "
    "'which stocks are up at least 15% this month'. Filtering is deterministic.",
    {
        "sector"    : {"type": "string", "description": "Sector name, e.g. 'technology'"},
        "period"    : {"type": "string", "default": "ytd", "description": "Period: '1mo','3mo','6mo','ytd','1y'"},
        "min_pct"   : {"type": "number", "default": 20.0, "description": "Minimum % return threshold (e.g. 20 means >=20%)"},
        "top_n"     : {"type": "integer", "default": 5, "description": "How many results to return"},
        "market_cap": {"type": "string", "enum": ["Large", "Mid", "Small", ""], "default": "Large"},
    }, ["sector"])

SCHEMA_RETURN_FILTER = _s("filter_sector_by_return_conditions",
    "Fetch ALL tickers for a sector, download returns for two periods, filter by sign conditions "
    "in Python, sort, and return top-N. Use this for any question like 'which stocks dropped this "
    "month but grew this year' or similar two-period filter. Filtering is deterministic and covers "
    "all tickers in the sector.",
    {
        "sector"    : {"type": "string", "description": "Sector name, e.g. 'technology'"},
        "period1"   : {"type": "string", "default": "1mo", "description": "First period: '1mo','3mo','6mo','ytd','1y'"},
        "condition1": {"type": "string", "enum": ["negative", "positive"], "default": "negative", "description": "Sign condition for period1"},
        "period2"   : {"type": "string", "default": "ytd", "description": "Second period: '1mo','3mo','6mo','ytd','1y'"},
        "condition2": {"type": "string", "enum": ["negative", "positive"], "default": "positive", "description": "Sign condition for period2"},
        "sort_by"   : {"type": "string", "enum": ["period1", "period2"], "default": "period2", "description": "Which period to sort by"},
        "top_n"     : {"type": "integer", "default": 3, "description": "How many results to return"},
        "market_cap": {"type": "string", "enum": ["Large", "Mid", "Small", ""], "default": "Large"},
    }, ["sector"])

ALL_SCHEMAS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW, SCHEMA_RANK, SCHEMA_52WEEK,
               SCHEMA_RETURN_FILTER, SCHEMA_MIN_RETURN, SCHEMA_RANK_RETURN, SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL]


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
- For sector/industry PRICE RETURN ranking ("top N by 1-year return", "best performers this month/year", etc.):
  Call rank_sector_by_return(sector=..., period=..., top_n=N).
  This tool handles DB lookup (works for sub-industries like 'semiconductor' too), downloads all returns, and sorts in Python. DO NOT call get_tickers_by_sector + get_price_performance manually for ranking.
- For sector/industry FUNDAMENTAL ranking ("top N stocks by P/E", "best by market cap", etc.):
  Call rank_stocks_by_metric(sector=..., metric=..., top_n=N, descending=True).
  This tool handles SQL lookup + data fetching + sorting in Python — the result is already correctly ordered. Report it as-is.
  DO NOT use get_tickers_by_sector or query_local_db for fundamental ranking — use rank_stocks_by_metric.
- For 52-WEEK LOW PROXIMITY questions ("closer to 52-week low", "near 52-week low", "trading near their lows"):
  Call filter_sector_by_52week(sector=...) FIRST. It returns only the qualifying stocks with current_price, week_high_52, week_low_52, and pct_above_low.
  Then call get_news_sentiment for each qualifying ticker (up to 5). DO NOT call get_company_overview separately — all 52-week data is already in the filter result.
- For other sector/industry questions (non-ranking): use get_tickers_by_sector or query_local_db to get tickers, THEN fetch data. Never guess tickers.
- For comparison questions: Fetch data for ALL tickers mentioned, then compare.
- For multi-condition questions (e.g., "dropped this month but grew this year"): Fetch data for BOTH time periods, then filter and compare.
- When a tool returns data, TRUST it and report it directly.
- If a tool returns an error or empty data, say so explicitly. Do not guess values when a tool fails.
- Present numerical results clearly with the data source identified.
- When ranking or filtering results, show the actual numbers you used to make the determination.
- Use the exact numerical values provided by the tools. DO NOT round or APPROXIMATE.

CONVERSATIONAL CONTEXT: If the user refers to "that", "it", "the two", or similar pronouns, use the conversation history provided to resolve the reference before fetching data.

OUTPUT FORMAT:
- For ranking questions ("top N by P/E", "best by market cap", etc.): one line per stock. FORMAT: N. TICKER (Company Name): P/E X.XX | EPS X.XX | Market Cap $X.XXB. No intro, no conclusion.
- For 52-week proximity questions: one block per stock. FORMAT: TICKER (Company Name): current $X.XX | 52-week $LOW - $HIGH | X.XX% above low. News: [headline label (score), ...]. Separate stocks with a blank line.
- For single-ticker questions: report the key metric(s) asked for plus 52-week range in 2-3 sentences max.
- For all other questions: answer concisely in plain prose, no bullet points, no markdown headers."""


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

MARKET_TOOLS      = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_RETURN_FILTER, SCHEMA_MIN_RETURN, SCHEMA_RANK_RETURN, SCHEMA_STATUS, SCHEMA_MOVERS]
FUNDAMENTAL_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_RANK, SCHEMA_52WEEK, SCHEMA_SQL, SCHEMA_TICKERS]
SENTIMENT_TOOLS   = [SCHEMA_NEWS, SCHEMA_52WEEK, SCHEMA_SQL]

ORCHESTRATOR_PROMPT = """You are a query router for a financial multi-agent system.
Given a user question, you decide which specialist agents to activate and write a focused sub-task for each one.

Specialists available:
- "Price"        — handles price performance, market status, top gainers/losers, sector tickers
- "Fundamentals" — handles P/E ratio, EPS, market cap, 52-week range, DB sector queries
- "Sentiment"    — handles news headlines and Bullish/Bearish/Neutral sentiment scores

RULES:
1. Only activate specialists that are STRICTLY needed to answer the question.
   - Questions about P/E, EPS, market cap, or 52-week range (without sentiment) → "Fundamentals" ONLY.
   - Questions about price change / performance → "Price" ONLY (unless also asking for fundamentals).
   - Questions about news or sentiment (with specific tickers) → "Sentiment" ONLY.
   - Only combine multiple specialists when the question explicitly asks for multiple domains in one answer.
2. Detect a cross-domain dependency and use phased execution:
   a) Price-then-fundamentals/sentiment: question ranks/filters by price/return then asks fundamentals or sentiment of top results → set "phased": true, "phase1_agent": "Price". Examples: "top 3 semiconductor stocks by 1-year return, what are their P/E ratios", "best energy stocks this year, show their news sentiment". Sub-task for Price MUST say exactly: "Call rank_sector_by_return(sector='[sector_from_question]', period='[period]', top_n=[N]). Report the tickers and returns from the results list directly."
   b) 52-week-then-sentiment: question asks which SECTOR stocks are closer to 52-week low AND also requests news sentiment → set "phased": true, "phase1_agent": "Fundamentals", agents: ["Fundamentals", "Sentiment"]. The Fundamentals agent identifies qualifying stocks; Sentiment receives those tickers as a hint.
3. Write a concise, self-contained sub-task string for each activated specialist. The sub-task must:
   - Include any ticker symbols mentioned in the original question.
   - Include the EXACT sector name from the original question verbatim (e.g. if the question says "finance sector", the sub-task must say "finance sector", not "financial sector", not "Financial Services", not any other substitution). The specialist agent resolves the canonical name itself.
4. Return ONLY valid JSON — no prose before or after.
5. Always activate at least one specialist — never return an empty "agents" list. If the question is purely about DB/sector lookup (e.g., "list companies in database"), activate "Price" (it has get_tickers_by_sector).
6. For questions requiring TWO time-period conditions on the SAME stocks (e.g., "dropped this month but grew this year", "fell recently but up this year"): this is ALWAYS a Price question. Set "phased": false, assign ONLY "Price". Sub-task template: "Call filter_sector_by_return_conditions(sector='[sector]', period1='1mo', condition1='negative', period2='ytd', condition2='positive', top_n=3). Report the results directly."
7. For questions asking which stocks in a sector have grown/gained/risen/increased by MORE THAN or AT LEAST a specific percentage over ONE period (e.g. "grown more than 20% this year", "up at least 15% this month", "gained over 30% YTD"): this is ALWAYS a Price question — NOT a Fundamentals question even if the question mentions large-cap or NASDAQ. Set "phased": false, assign ONLY "Price". Sub-task template: "Call filter_sector_by_min_return(sector='[sector]', period='[ytd/1mo/etc]', min_pct=[threshold], top_n=[N]). Report the results directly."

Return format:
{
  "agents": ["Fundamentals"],
  "phased": false,
  "phase1_agent": "",
  "task_per_agent": {
    "Fundamentals": "..."
  }
}"""

PRICE_AGENT_PROMPT = """You are a market-data specialist with access to price performance, market status, top movers, and sector/ticker lookup tools.
Answer accurately using only the data your tools return. If a tool fails, say so.
Do not guess ticker symbols — use get_tickers_by_sector to look them up first when needed.

RANKING PROTOCOL — ALWAYS use this when asked to find top-N stocks in a sector/industry by price return:
1. Call rank_sector_by_return(sector=<sector_from_task>, period=<period>, top_n=N).
   This tool handles DB lookup (including sub-industries like 'semiconductor'), downloads returns for ALL tickers, and sorts in Python. The result is already correctly ordered.
2. Report the tickers and returns DIRECTLY from the "results" list. Use the exact "ticker" and period values from the JSON.
3. DO NOT call get_tickers_by_sector or get_price_performance manually for any top-N ranking question — rank_sector_by_return does it all.

STRICT TICKER RULES:
- The KEY in the get_price_performance response IS the ticker. Its value belongs only to that ticker.
- Never assign one ticker's pct_change to a different ticker symbol.
- Never infer or guess a return value from your training knowledge.

OUTPUT FORMAT:
A) COMPARISON (multiple specific tickers): Report ALL requested tickers: TICKER: start=$X.XX  end=$X.XX  change=+Y.YY%
B) SINGLE TICKER: TICKER: start=$X.XX  end=$X.XX  change=+Y.YY%
C) SECTOR RANKING (top-N): List top-N from sorted scratchpad: 1. TICKER: +X.XX%

MULTI-CONDITION FILTERING PROTOCOL — for two-period filter questions (e.g., "dropped this month but grew this year"):
1. Call filter_sector_by_return_conditions(sector=..., period1=..., condition1=..., period2=..., condition2=..., top_n=N).
   This tool fetches ALL sector tickers, downloads both periods, filters and sorts in Python — the result is already correct.
2. Report the results directly using the EXACT float values from the tool's "results" list. For each entry, output one line:
   TICKER (Company Name): 1mo [period1 value]% | YTD [period2 value]%
   where [period1 value] and [period2 value] are the actual numbers returned in the JSON (e.g. -3.21, +18.44), and Company Name comes from the "name" field. NEVER write placeholder text.
3. Do NOT call get_tickers_by_sector or get_price_performance manually for this type of question.

SINGLE-PERIOD THRESHOLD FILTER PROTOCOL — for questions like "which stocks grew more than 20% this year" or "up at least 15% this month":
1. Call filter_sector_by_min_return(sector=..., period=..., min_pct=..., top_n=N).
   This tool fetches ALL sector tickers, downloads the period, filters >= min_pct in Python, and returns sorted results.
2. Report the results directly using the EXACT float values from the tool's "results" list. For each entry, output one line:
   TICKER (Company Name): [period] [value]%
   using the actual number from the JSON (e.g. +42.67) and the "name" field for the company name. NEVER write placeholder text.
3. Do NOT call get_tickers_by_sector or get_price_performance manually for this type of question.

Always report the exact numeric pct_change for every stock you mention."""

FUNDAMENTALS_AGENT_PROMPT = """You are a fundamental analysis specialist with access to company overview data (P/E, EPS, market cap, 52-week range) and a local stocks database.
Answer accurately using only the data your tools return. If a tool fails, say so.
Present all values clearly with the field name (e.g. "P/E ratio: 28.5").

SECTOR RANKING PROTOCOL — follow these steps EXACTLY when asked to rank/find top stocks in a sector by P/E or any other metric:
1. Call rank_stocks_by_metric(sector=..., metric="pe_ratio", top_n=5, descending=True).
   This handles SQL lookup + data fetching + Python sort internally. The result is already correctly sorted.
2. Report the ranked list from the result directly.
IMPORTANT: Use rank_stocks_by_metric for ALL sector ranking questions. Do NOT call query_local_db + get_company_overview manually for ranking.

52-WEEK LOW PROXIMITY PROTOCOL — follow these steps EXACTLY when asked which stocks are closer to their 52-week low:
1. Identify the sector name from your task. Pass it EXACTLY as written (e.g. if task says "finance sector", call filter_sector_by_52week(sector="finance")). Do NOT substitute a different sector name from your training knowledge.
2. Call filter_sector_by_52week(sector=<sector from task>) ONCE.
   It returns qualifying stocks with ticker, name, current_price, week_high_52, week_low_52, and pct_above_low, sorted closest-to-low first.
3. Report EVERY qualifying stock in this exact format per line:
   TICKER (Company Name): current $X.XX | 52-week $LOW - $HIGH | X.XX% above low
   Use the EXACT "ticker" field value from the tool result (e.g. "BRO", "BX"). NEVER substitute a company name or abbreviation as a ticker.
   Use the "name" field for the company name in parentheses.
   DO NOT call get_company_overview separately.
IMPORTANT: Use filter_sector_by_52week for ALL 52-week low proximity questions."""

SENTIMENT_AGENT_PROMPT = """You are a news and sentiment specialist with access to real-time news headlines and sentiment scores for individual stocks.
Summarise sentiment clearly: Bullish / Bearish / Neutral with score.
If a tool fails or returns no articles, say so explicitly. Do not fabricate headlines.
IMPORTANT: Only fetch sentiment for the specific tickers listed in your task.
If your task requests sentiment for a sector but gives no specific tickers, first call filter_sector_by_52week(sector=...) to get the qualifying ticker list, then call get_news_sentiment for each ticker returned."""

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

1. TABLE FORMAT — use ONLY when ALL of these are true:
   a) price_returns contains 2 or more tickers, AND
   b) the original_question explicitly asks to COMPARE price % change across tickers (e.g. "which gained most", "top performers", "best/worst performing", "price change").
   DO NOT use TABLE FORMAT for: 52-week range questions, P/E questions, sentiment questions, or any MULTI-CONDITION FILTER question (e.g. "dropped this month but grew this year", "fell recently but up this year"). Those use PROSE FORMAT.
   - Output exactly one line per ticker. FORMAT: TICKER: +X.XX% | P/E X.XX | Sentiment: Label (score)
   - Use the exact float from price_returns for %. Prefix positive with +.
   - Get P/E and Sentiment from specialist_answers. Write N/A only if truly absent.
   - No company names, no intro sentence, no conclusion, no blank lines.

2. RANKING FORMAT — use ONLY when the question asks to rank stocks by a FUNDAMENTAL metric (P/E, EPS, market cap, dividend yield, etc.) and price_returns is empty.
   DO NOT use RANKING FORMAT for price-return rankings ("most gained", "top performers", "dropped this month", "grew this year", "top N by return") — use PROSE FORMAT for those instead.
   - One line per stock. FORMAT: N. TICKER (Company Name): P/E X.XX | EPS X.XX | Market Cap $X.XXB
   - Always include the company name in parentheses after the ticker. If not available, omit the parentheses.
   - Include only the fields the question asks about plus any extra fields present in the specialist answer. Omit fields that are "None" or "N/A".
   - No intro sentence, no conclusion, no blank lines between entries.

3. PROSE FORMAT — use for all other questions:
   - Include ALL key numeric values from the specialist answers.
   - For SENTIMENT: list EVERY headline with its exact label and score.
   - For SINGLE-TICKER PRICE: report start, end, AND % change.
   - For 52-WEEK RANGE FILTER: output EXACTLY ONE LINE per stock, then a blank line before the next stock. Format per stock:
     TICKER (Company Name): current $X.XX | 52-week $LOW - $HIGH | X.XX% above low | Sentiment: Label (score), Label (score), ...
     Always include the company name in parentheses — it is present in the specialist answer. Every stock MUST start on its own new line. No two stocks on the same line.
   - For MULTI-CONDITION PRICE FILTER ("dropped this month but grew this year", "top N by return", etc.): list every qualifying stock with BOTH actual numeric values from the tool output. Each stock on its own line. Example (using made-up numbers): "1. AAPL (Apple Inc.): 1mo -3.21% | YTD +15.44%". Use the "name" field from the tool result for the company name in parentheses. Replace the numbers with the EXACT values returned by the tool — do NOT write placeholder text like X.XX. If the answer says no stocks qualified, relay that verbatim.
   - For SINGLE-PERIOD THRESHOLD FILTER ("grown more than X% this year", "up at least X%", "gained over X%"): list every qualifying stock with its actual numeric return. Each stock on its own line. Example (using made-up numbers): "1. AAPL (Apple Inc.): YTD +38.42%". Use the "name" field from the tool result for the company name in parentheses. Replace the numbers with the EXACT values from the specialist answer — do NOT write placeholder text.
   - Draw facts from specialist_answers only.
   - No markdown, no bullet points, no headers.
   - Use numerical values exactly as provided.
   - If a required data field is missing from all specialist answers, say so inline briefly."""


def run_orchestrator(question: str) -> dict:
    """One LLM call to plan which specialists to activate."""
    response = client.chat.completions.create(
        model=ACTIVE_MODEL,
        temperature=0,
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
        temperature=0,
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
        temperature=0,
        messages=[
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {"role": "user",   "content": json.dumps(payload)},
        ],
    )
    return response.choices[0].message.content or ""


def _extract_tickers_from_result(result: AgentResult, top_n: int = 3) -> list:
    tickers = []
    seen = set()
    for tool_output in result.raw_data.values():
        if not isinstance(tool_output, dict):
            continue
        # Price agent format: {TICKER: {"pct_change": float}}
        for k, v in tool_output.items():
            if (isinstance(k, str) and k.isupper() and 1 <= len(k) <= 5
                    and isinstance(v, dict) and "pct_change" in v
                    and k not in seen):
                tickers.append((k, float(v["pct_change"])))
                seen.add(k)
        # filter_sector_by_52week format: {"qualifying": [{"ticker": ..., "pct_above_low": float}]}
        if "qualifying" in tool_output:
            for item in tool_output["qualifying"]:
                t = item.get("ticker", "")
                if t and t not in seen:
                    tickers.append((t, -(item.get("pct_above_low") or 999)))
                    seen.add(t)
        # rank_sector_by_return / filter_sector_by_min_return / filter_sector_by_return_conditions
        # format: {"results": [{"ticker": ..., "name": ..., period: "+XX.XX"}]}
        if "results" in tool_output:
            for item in tool_output["results"]:
                t = item.get("ticker", "")
                if t and t not in seen:
                    # Find numeric value from whichever period key is present
                    pct_val = 0.0
                    for key, val in item.items():
                        if key not in ("ticker", "name") and isinstance(val, str):
                            try:
                                pct_val = float(val)
                                break
                            except ValueError:
                                pass
                    tickers.append((t, pct_val))
                    seen.add(t)
    tickers.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in tickers[:top_n]]


def run_multi_agent(question: str, verbose: bool = False) -> dict:
    """Orchestrator → Parallel Specialists → Critic → Synthesizer pipeline."""
    t0 = time.time()
    plan = run_orchestrator(question)

    # Enable rate limiting for the duration of this function so parallel
    # specialist threads don't burst-hammer Yahoo Finance.
    try:
        import av_mock_server as _avms
        _avms.set_rate_limiting(True)
    except Exception:
        _avms = None

    agent_results: list[AgentResult] = []
    price_returns: dict = {}

    if plan.get("phased") and plan.get("phase1_agent"):
        p1_name = plan["phase1_agent"]
        p1_task = plan["task_per_agent"].get(p1_name, question)
        p1_result = SPECIALIST_RUNNERS[p1_name](p1_task, verbose=verbose)
        agent_results.append(p1_result)

        _p1_top_n = 8 if p1_name == "Fundamentals" else 3
        top_tickers = _extract_tickers_from_result(p1_result, top_n=_p1_top_n)
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

    # All parallel work is done — disable rate limiting immediately so
    # subsequent single-agent calls are not slowed down.
    if _avms is not None:
        _avms.set_rate_limiting(False)

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
