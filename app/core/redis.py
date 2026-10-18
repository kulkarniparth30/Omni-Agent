import logging
import json
from typing import Optional, Tuple, Any
import redis.asyncio as aioredis
from app.config import get_settings

logger = logging.getLogger("app.core.redis")

_REDIS_CLIENT: Optional[aioredis.Redis] = None
_REDIS_AVAILABLE: bool = False

async def init_redis() -> Optional[aioredis.Redis]:
    """
    Initialize an asynchronous Redis connection pool.
    Gracefully degrades if Redis is unavailable, so the app remains 100% functional
    even in environments without a running Redis instance.
    """
    global _REDIS_CLIENT, _REDIS_AVAILABLE
    settings = get_settings()

    if not settings.REDIS_ENABLED or not settings.REDIS_URL:
        logger.info("Redis is disabled via configuration.")
        _REDIS_AVAILABLE = False
        return None

    try:
        client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0
        )
        # Test connection ping
        await client.ping()
        _REDIS_CLIENT = client
        _REDIS_AVAILABLE = True
        logger.info("Connected successfully to Redis at '%s'.", settings.REDIS_URL)
        return _REDIS_CLIENT
    except Exception as exc:
        logger.warning(
            "Redis connection to '%s' failed (%s). Continuing in fallback mode (in-memory/no-op).",
            settings.REDIS_URL, exc
        )
        _REDIS_AVAILABLE = False
        _REDIS_CLIENT = None
        return None

async def close_redis() -> None:
    """Cleanly close the Redis connection pool during application shutdown."""
    global _REDIS_CLIENT, _REDIS_AVAILABLE
    if _REDIS_CLIENT:
        try:
            await _REDIS_CLIENT.aclose()
            logger.info("Closed Redis connection pool.")
        except Exception as exc:
            logger.warning("Error closing Redis pool: %s", exc)
    _REDIS_CLIENT = None
    _REDIS_AVAILABLE = False

def is_redis_available() -> bool:
    """Return whether Redis is currently connected and ready."""
    return _REDIS_AVAILABLE and _REDIS_CLIENT is not None

async def check_rate_limit(
    client_identifier: str,
    limit: Optional[int] = None,
    window_seconds: int = 60
) -> Tuple[bool, int, int]:
    """
    Sliding/fixed-window atomic rate limiter via Redis.
    
    Args:
        client_identifier: IP address, API key, or user identifier.
        limit: Max requests allowed in window (defaults to settings.RATE_LIMIT_PER_MINUTE).
        window_seconds: Window duration in seconds (defaults to 60s).
        
    Returns:
        (is_allowed: bool, current_usage: int, retry_after_seconds: int)
    """
    settings = get_settings()
    max_limit = limit or settings.RATE_LIMIT_PER_MINUTE

    if not is_redis_available() or _REDIS_CLIENT is None:
        # Fallback: allow all requests if Redis is offline
        return True, 0, 0

    key = f"rate_limit:{client_identifier}"
    try:
        async with _REDIS_CLIENT.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.ttl(key)
            results = await pipe.execute()

        current_count = results[0]
        ttl = results[1]

        # Set initial expiry if this is the first hit in the window
        if ttl == -1 or current_count == 1:
            await _REDIS_CLIENT.expire(key, window_seconds)
            ttl = window_seconds

        if current_count > max_limit:
            retry_after = max(1, ttl)
            return False, current_count, retry_after

        return True, current_count, 0
    except Exception as exc:
        logger.warning("Rate limit check failed against Redis (%s). Failing open.", exc)
        return True, 0, 0

async def get_cached_result(cache_key: str) -> Optional[Any]:
    """Fetch cached data from Redis. Returns None if cache miss or Redis offline."""
    if not is_redis_available() or _REDIS_CLIENT is None:
        return None

    try:
        raw = await _REDIS_CLIENT.get(f"cache:{cache_key}")
        if raw is not None:
            return json.loads(raw)
    except Exception as exc:
        logger.warning("Cache GET failed for key '%s': %s", cache_key, exc)
    return None

async def set_cached_result(cache_key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
    """Store data in Redis with a Time-To-Live (TTL)."""
    if not is_redis_available() or _REDIS_CLIENT is None:
        return False

    settings = get_settings()
    ttl = ttl_seconds or settings.TOOL_CACHE_TTL_SECONDS

    try:
        payload = json.dumps(value)
        await _REDIS_CLIENT.set(f"cache:{cache_key}", payload, ex=ttl)
        return True
    except Exception as exc:
        logger.warning("Cache SET failed for key '%s': %s", cache_key, exc)
        return False
