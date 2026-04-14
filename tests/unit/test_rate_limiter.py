from mcp_server.security import RateLimitConfig, SlidingWindowRateLimiter


def test_rate_limiter_blocks_after_threshold() -> None:
    limiter = SlidingWindowRateLimiter(RateLimitConfig(requests_per_window=2, window_seconds=10))
    key = "127.0.0.1"

    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False
