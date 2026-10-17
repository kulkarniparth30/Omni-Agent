import requests
from langchain_core.tools import tool
from app.config import get_settings

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price and quote for a given ticker symbol (e.g. 'AAPL', 'TSLA')
    using the Alpha Vantage API.
    """
    settings = get_settings()
    api_key = settings.ALPHA_VANTAGE_API_KEY
    if not api_key:
        return {
            "error": "ALPHA_VANTAGE_API_KEY is not configured.",
            "symbol": symbol
        }

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol.strip().upper()}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "error": f"Failed to retrieve stock quote: {str(e)}",
            "symbol": symbol
        }
