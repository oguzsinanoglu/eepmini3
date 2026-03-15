"""
Mock Alpha Vantage API Server
Uses yfinance as data source. Falls back to random plausible data if yfinance fails.

Usage:
    1. pip install flask yfinance pytz websockets beautifulsoup4
    2. python av_mock_server.py
    3. In notebook, add after load_dotenv():
    
       AV_BASE = "http://localhost:2345"
       
    4. In each tool function that calls Alpha Vantage, replace:
         "https://www.alphavantage.co"  ->  AV_BASE
"""

from flask import Flask, request, jsonify
import yfinance as yf
import requests as _req_module
import threading
import random
import time
import traceback
from datetime import datetime
from pytz import timezone

# ── Shared yfinance session (preserves cookies/crumb across calls) ────────────
_SESSION_LOCK = threading.Lock()
_SHARED_YF_SESSION = None

def _get_shared_session():
    """Return (or lazily create) a single shared session for all yfinance calls.
    curl_cffi impersonates Chrome's full TLS fingerprint. Falls back to
    requests.Session with a browser User-Agent if curl_cffi is unavailable."""
    global _SHARED_YF_SESSION
    if _SHARED_YF_SESSION is not None:
        return _SHARED_YF_SESSION
    with _SESSION_LOCK:
        if _SHARED_YF_SESSION is None:  # double-checked locking
            try:
                from curl_cffi import requests as _curl_requests
                _SHARED_YF_SESSION = _curl_requests.Session(impersonate="chrome120")
            except ImportError:
                s = _req_module.Session()
                s.headers.update({
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    ),
                })
                _SHARED_YF_SESSION = s
    return _SHARED_YF_SESSION

def _yf_ticker(symbol: str) -> yf.Ticker:
    """Return a yfinance Ticker backed by the shared session."""
    return yf.Ticker(symbol, session=_get_shared_session())

# ── Thread-safe rate limiter for Yahoo Finance calls ─────────────────────────
_YF_RATE_LOCK = threading.Lock()
_LAST_YF_CALL_TIME = 0.0
_MIN_YF_CALL_GAP   = 0.4  # seconds between calls; keeps bursts under Yahoo's threshold

# Rate limiting is OFF by default (single-agent is sequential and doesn't need it).
# Enable it before running concurrent multi-agent specialists, then disable again.
_RATE_LIMIT_ENABLED = False

def set_rate_limiting(enabled: bool) -> None:
    """Toggle the inter-call rate limiter. Call with True before parallel
    multi-agent execution and False immediately after."""
    global _RATE_LIMIT_ENABLED
    _RATE_LIMIT_ENABLED = enabled

app = Flask(__name__)

_INFO_CACHE_TTL = 300  # seconds; re-fetch after 5 minutes or on failure
_info_cache: dict = {}  # ticker -> (fetched_at: float, data: dict | None)

def _get_info(ticker):
    """Get fundamentals for one ticker with TTL caching.

    Strategy order (no blocking delays anywhere):
    1. Cache hit — return immediately.
    2. Targeted HTTP quoteSummary: requests ONLY summaryDetail+defaultKeyStatistics.
       The shared session already has Yahoo cookies from prior fast_info calls, so
       this lighter 2-module request succeeds even when the full multi-module
       ticker.info is rate-limited on cloud IPs.
    3. Full ticker.info — one quick attempt, no sleep.
    4. Returns None; _handle_overview falls back to fast_info for the basic fields.
    """
    global _LAST_YF_CALL_TIME
    entry = _info_cache.get(ticker)
    if entry is not None:
        fetched_at, data = entry
        if time.time() - fetched_at < _INFO_CACHE_TTL:
            return data

    # Rate limiting only during concurrent multi-agent parallel execution
    if _RATE_LIMIT_ENABLED:
        with _YF_RATE_LOCK:
            elapsed = time.time() - _LAST_YF_CALL_TIME
            if elapsed < _MIN_YF_CALL_GAP:
                time.sleep(_MIN_YF_CALL_GAP - elapsed)
            _LAST_YF_CALL_TIME = time.time()

    sess = _get_shared_session()
    t    = _yf_ticker(ticker)

    # Strategy 1: targeted 2-module quoteSummary via shared session
    _QS_URL = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    try:
        r = sess.get(
            _QS_URL,
            params={"modules": "summaryDetail,defaultKeyStatistics", "formatted": "false"},
            headers={"Accept": "application/json"},
            timeout=8,
        )
        if r.status_code == 200:
            parts = (r.json().get("quoteSummary") or {}).get("result") or []
            if parts:
                combined = {}
                for part in parts:
                    combined.update(part)
                sd  = combined.get("summaryDetail", {})
                ks  = combined.get("defaultKeyStatistics", {})
                pe  = sd.get("trailingPE") or sd.get("forwardPE")
                eps = ks.get("trailingEps") or ks.get("forwardEps")
                if pe is not None:
                    fi = t.fast_info
                    info = {
                        "shortName"       : ticker,
                        "sector"          : "N/A",
                        "industry"        : "N/A",
                        "trailingPE"      : sd.get("trailingPE"),
                        "forwardPE"       : sd.get("forwardPE"),
                        "trailingEps"     : eps,
                        "marketCap"       : getattr(fi, "market_cap",         None),
                        "fiftyTwoWeekHigh": getattr(fi, "fifty_two_week_high", None),
                        "fiftyTwoWeekLow" : getattr(fi, "fifty_two_week_low",  None),
                        "beta"            : sd.get("beta"),
                        "dividendYield"   : sd.get("dividendYield"),
                    }
                    _info_cache[ticker] = (time.time(), info)
                    return info
    except Exception:
        pass

    # Strategy 2: full ticker.info — one quick attempt, no delay
    try:
        data = t.get_info() if hasattr(t, "get_info") else t.info
        if data and data.get("shortName"):
            _info_cache[ticker] = (time.time(), data)
            return data
    except Exception:
        pass

    return None  # not cached — will be retried on next call


def _is_market_open(tz_name, open_h, open_m, close_h, close_m):
    """Check if current time in given timezone is within trading hours on a weekday."""
    now = datetime.now(timezone(tz_name))
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = now.hour * 60 + now.minute
    return (open_h * 60 + open_m) <= t < (close_h * 60 + close_m)


def _handle_overview(params):
    ticker = params.get("symbol", "")
    if not ticker:
        return {}

    info = _get_info(ticker)

    def safe(val):
        return "None" if val is None else str(val)

    if info and info.get("shortName"):
        pe = info.get("trailingPE") or info.get("forwardPE")
        eps = info.get("trailingEps") or info.get("forwardEps")
        return {
            "Symbol": ticker,
            "Name": info.get("shortName", ticker),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "MarketCapitalization": safe(info.get("marketCap")),
            "PERatio": safe(pe),
            "EPS": safe(eps),
            "52WeekHigh": safe(info.get("fiftyTwoWeekHigh")),
            "52WeekLow": safe(info.get("fiftyTwoWeekLow")),
            "DividendYield": safe(info.get("dividendYield")),
            "Beta": safe(info.get("beta")),
        }

    # Fallback: fast_info hits a lighter Yahoo endpoint, less rate-limited
    try:
        fi = _yf_ticker(ticker).fast_info
        return {
            "Symbol": ticker,
            "Name": ticker,
            "Sector": "N/A",
            "Industry": "N/A",
            "MarketCapitalization": safe(getattr(fi, "market_cap", None)),
            "PERatio": "None",
            "EPS": "None",
            "52WeekHigh": safe(getattr(fi, "fifty_two_week_high", None)),
            "52WeekLow": safe(getattr(fi, "fifty_two_week_low", None)),
            "DividendYield": "None",
            "Beta": "None",
        }
    except Exception:
        return {}


def _handle_market_status(params):
    us_open = _is_market_open("US/Eastern", 9, 30, 16, 15)
    uk_open = _is_market_open("Europe/London", 8, 0, 16, 30)
    jp_open = _is_market_open("Asia/Tokyo", 9, 0, 15, 0)

    return {
        "endpoint": "Global Market Open & Close Status",
        "markets": [
            {
                "market_type": "Equity",
                "region": "United States",
                "primary_exchanges": "NYSE, NASDAQ, AMEX, BATS",
                "local_open": "09:30",
                "local_close": "16:15",
                "current_status": "open" if us_open else "closed",
                "notes": ""
            },
            {
                "market_type": "Equity",
                "region": "United Kingdom",
                "primary_exchanges": "London Stock Exchange",
                "local_open": "08:00",
                "local_close": "16:30",
                "current_status": "open" if uk_open else "closed",
                "notes": ""
            },
            {
                "market_type": "Equity",
                "region": "Japan",
                "primary_exchanges": "Tokyo Stock Exchange",
                "local_open": "09:00",
                "local_close": "15:00",
                "current_status": "open" if jp_open else "closed",
                "notes": ""
            },
        ]
    }


def _handle_top_gainers_losers(params):
    """Scrape Yahoo Finance for real data, fall back to random."""
    try:
        from bs4 import BeautifulSoup
        import requests as req

        def scrape_yahoo(url, n=5):
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = req.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("table tbody tr")
            results = []
            for row in rows[:n]:
                cells = row.select("td")
                if len(cells) < 5:
                    continue
                results.append({
                    "ticker": cells[0].get_text(strip=True),
                    "price": cells[3].get_text(strip=True),
                    "change_amount": cells[4].get_text(strip=True),
                    "change_percentage": cells[5].get_text(strip=True) if len(cells) > 5 else "",
                    "volume": cells[6].get_text(strip=True) if len(cells) > 6 else "",
                })
            return results

        gainers = scrape_yahoo("https://finance.yahoo.com/markets/stocks/gainers/")
        losers  = scrape_yahoo("https://finance.yahoo.com/markets/stocks/losers/")
        active  = scrape_yahoo("https://finance.yahoo.com/markets/stocks/most-active/")

        if not gainers and not losers and not active:
            raise ValueError("Scrape returned empty")

        return {
            "metadata": "Top Gainers, Losers, and Most Active (yahoo scrape)",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "top_gainers": gainers or [],
            "top_losers": losers or [],
            "most_actively_traded": active or [],
        }
    except Exception as e:
        print(f"  [mock-av] yahoo scrape failed ({e}), using random fallback")
        return _handle_top_gainers_losers_fallback()


def _handle_top_gainers_losers_fallback():
    """Random fallback when yahoo_fin is unavailable."""
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL",
               "AMD", "INTC", "NFLX", "CRM", "ORCL", "PYPL", "SQ", "SHOP",
               "PLTR", "RIVN", "LCID", "NIO", "SOFI"]

    def random_movers(n=5):
        picked = random.sample(tickers, n)
        results = []
        for t in picked:
            price = round(random.uniform(10, 400), 2)
            change = round(random.uniform(2, 15), 2)
            results.append({
                "ticker": t,
                "price": str(price),
                "change_amount": str(round(price * change / 100, 2)),
                "change_percentage": f"{change}%",
                "volume": str(random.randint(1000000, 50000000))
            })
        return results

    return {
        "metadata": "Top Gainers, Losers, and Most Active (random fallback)",
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_gainers": random_movers(),
        "top_losers": random_movers(),
        "most_actively_traded": random_movers(),
    }


def _handle_news_sentiment(params):
    ticker = params.get("tickers", "")
    limit = int(params.get("limit", 5))

    articles = []

    try:
        news = _yf_ticker(ticker).news
        if news:
            for item in news[:limit]:
                content = item.get("content", {})
                title = content.get("title", "")
                provider = content.get("provider", {})
                source = provider.get("displayName", "Unknown")

                if not title:
                    continue

                sent_options = ["Bullish", "Somewhat-Bullish", "Neutral",
                                "Somewhat-Bearish", "Bearish"]
                weights = [0.15, 0.3, 0.3, 0.15, 0.1]
                sent = random.choices(sent_options, weights=weights, k=1)[0]
                score_map = {
                    "Bullish": random.uniform(0.3, 0.6),
                    "Somewhat-Bullish": random.uniform(0.1, 0.35),
                    "Neutral": random.uniform(-0.1, 0.1),
                    "Somewhat-Bearish": random.uniform(-0.35, -0.1),
                    "Bearish": random.uniform(-0.6, -0.3),
                }
                articles.append({
                    "title": title,
                    "source": source,
                    "overall_sentiment_label": sent,
                    "overall_sentiment_score": str(round(score_map[sent], 6)),
                    "time_published": content.get("pubDate", time.strftime("%Y%m%dT%H%M%S")),
                })
    except Exception:
        pass

    fake_headlines = [
        f"{ticker} Shows Strong Momentum Amid Market Volatility",
        f"Analysts Upgrade {ticker} Price Target Following Earnings Beat",
        f"{ticker} Faces Headwinds From Regulatory Concerns",
        f"Institutional Investors Increase Holdings in {ticker}",
        f"{ticker} Announces Strategic Partnership in AI Sector",
        f"Market Watch: {ticker} Trading Volume Surges",
        f"{ticker} Q4 Results Exceed Wall Street Expectations",
        f"Why {ticker} Could Be a Top Pick for Growth Investors",
        f"{ticker} Expands Into New Markets With Latest Acquisition",
    ]
    random.shuffle(fake_headlines)

    idx = 0
    while len(articles) < limit:
        sent = random.choice(["Bullish", "Somewhat-Bullish", "Neutral",
                              "Somewhat-Bearish", "Bearish"])
        articles.append({
            "title": fake_headlines[idx % len(fake_headlines)],
            "source": random.choice(["Reuters", "Bloomberg", "Yahoo Finance",
                                     "MarketWatch", "CNBC", "Seeking Alpha"]),
            "overall_sentiment_label": sent,
            "overall_sentiment_score": str(round(random.uniform(-0.5, 0.5), 6)),
            "time_published": time.strftime("%Y%m%dT%H%M%S"),
        })
        idx += 1

    return {
        "items": str(len(articles[:limit])),
        "sentiment_score_definition": {},
        "feed": articles[:limit],
    }


@app.route("/query", methods=["GET"])
def handle_query():
    function = request.args.get("function", "")
    symbol = request.args.get("symbol", request.args.get("tickers", ""))
    print(f"  [mock-av] {function} symbol={symbol}")

    try:
        if function == "OVERVIEW":
            return jsonify(_handle_overview(request.args))
        elif function == "MARKET_STATUS":
            return jsonify(_handle_market_status(request.args))
        elif function == "TOP_GAINERS_LOSERS":
            return jsonify(_handle_top_gainers_losers(request.args))
        elif function == "NEWS_SENTIMENT":
            return jsonify(_handle_news_sentiment(request.args))
        else:
            return jsonify({"error": f"Unknown function: {function}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("=" * 50)
    print("  Mock Alpha Vantage Server")
    print("  http://localhost:2345")
    print("=" * 50)
    print()
    print("Notebook usage:")
    print('  AV_BASE = "http://localhost:2345"')
    print('  # then replace "https://www.alphavantage.co" with AV_BASE')
    print()
    app.run(host="0.0.0.0", port=2345, debug=False)