import requests
from langchain_core.tools import tool
from app.config import get_settings

import time
from typing import Dict, Tuple

# In-memory L1 cache: symbol -> (timestamp, data)
_STOCK_CACHE: Dict[str, Tuple[float, dict]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price and quote for a given ticker symbol (e.g. 'AAPL', 'TSLA')
    using the Alpha Vantage API. Cached for 5 minutes to prevent rate limit exhaustion.
    """
    clean_sym = symbol.strip().upper()
    now = time.time()

    # Check L1 cache
    if clean_sym in _STOCK_CACHE:
        cached_time, cached_data = _STOCK_CACHE[clean_sym]
        if now - cached_time < CACHE_TTL_SECONDS:
            return {**cached_data, "_cached": True, "_cached_age_seconds": round(now - cached_time, 1)}

    settings = get_settings()
    api_key = settings.ALPHA_VANTAGE_API_KEY
    if not api_key:
        return {
            "error": "ALPHA_VANTAGE_API_KEY is not configured.",
            "symbol": clean_sym
        }

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={clean_sym}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Only cache valid quotes
        if "Global Quote" in data and data["Global Quote"]:
            _STOCK_CACHE[clean_sym] = (now, data)

        return data
    except Exception as e:
        return {
            "error": f"Failed to retrieve stock quote: {str(e)}",
            "symbol": clean_sym
        }
