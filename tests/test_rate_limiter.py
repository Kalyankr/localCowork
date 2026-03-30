"""Tests for the rate limiter middleware."""

import time
from unittest.mock import MagicMock

import pytest


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def _make_limiter(self, max_requests=5, window_seconds=10):
        from agent.orchestrator.middleware import RateLimiter

        return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    def test_allows_requests_under_limit(self):
        rl = self._make_limiter(max_requests=3)
        assert rl.is_allowed("client-1") is True
        assert rl.is_allowed("client-1") is True
        assert rl.is_allowed("client-1") is True

    def test_blocks_after_limit_exceeded(self):
        rl = self._make_limiter(max_requests=2)
        rl.is_allowed("c1")
        rl.is_allowed("c1")
        assert rl.is_allowed("c1") is False

    def test_different_clients_are_independent(self):
        rl = self._make_limiter(max_requests=1)
        assert rl.is_allowed("a") is True
        assert rl.is_allowed("b") is True
        # 'a' is blocked, 'b' still good within its own limit
        assert rl.is_allowed("a") is False
        assert rl.is_allowed("b") is False

    def test_window_expiry_resets_counter(self):
        rl = self._make_limiter(max_requests=1, window_seconds=0)
        # With 0-second window all old requests are immediately expired
        assert rl.is_allowed("c") is True
        # Manually inject an old timestamp so the window cleans it
        rl.requests["c"] = [time.time() - 100]
        assert rl.is_allowed("c") is True  # Old one cleaned, new allowed

    def test_get_retry_after_returns_positive_seconds(self):
        rl = self._make_limiter(max_requests=1, window_seconds=60)
        rl.is_allowed("c")
        retry = rl.get_retry_after("c")
        assert 0 < retry <= 60

    def test_get_retry_after_returns_zero_for_unknown_client(self):
        rl = self._make_limiter()
        assert rl.get_retry_after("unknown") == 0

    def test_burst_then_wait(self):
        rl = self._make_limiter(max_requests=3, window_seconds=1)
        for _ in range(3):
            rl.is_allowed("c")
        assert rl.is_allowed("c") is False
        # Simulate time passing beyond the window
        rl.requests["c"] = [time.time() - 2 for _ in rl.requests["c"]]
        assert rl.is_allowed("c") is True

    def test_remaining_shows_quota(self):
        rl = self._make_limiter(max_requests=5)
        assert rl.remaining("c") == 5
        rl.is_allowed("c")
        rl.is_allowed("c")
        assert rl.remaining("c") == 3

    def test_remaining_zero_when_exhausted(self):
        rl = self._make_limiter(max_requests=2)
        rl.is_allowed("c")
        rl.is_allowed("c")
        assert rl.remaining("c") == 0

    def test_reset_single_client(self):
        rl = self._make_limiter(max_requests=1)
        rl.is_allowed("a")
        rl.is_allowed("b")
        assert rl.is_allowed("a") is False
        rl.reset("a")
        assert rl.is_allowed("a") is True
        # b should still be exhausted
        assert rl.is_allowed("b") is False

    def test_reset_all_clients(self):
        rl = self._make_limiter(max_requests=1)
        rl.is_allowed("a")
        rl.is_allowed("b")
        rl.reset()  # clear all
        assert rl.is_allowed("a") is True
        assert rl.is_allowed("b") is True


class TestResolveRateLimitKey:
    """resolve_rate_limit_key should prefer API key > session > IP."""

    def _make_request(self, headers=None, cookies=None, client_host="127.0.0.1"):
        from agent.orchestrator.middleware import resolve_rate_limit_key

        req = MagicMock()
        req.headers = headers or {}
        req.cookies = cookies or {}
        req.client = MagicMock()
        req.client.host = client_host
        return req, resolve_rate_limit_key

    def test_uses_api_key_when_present(self):
        req, resolve = self._make_request(headers={"x-api-key": "abc123"})
        key = resolve(req)
        assert key == "key:abc123"

    def test_uses_session_cookie_when_no_api_key(self):
        req, resolve = self._make_request(cookies={"session_id": "sess-42"})
        key = resolve(req)
        assert key == "session:sess-42"

    def test_falls_back_to_ip(self):
        req, resolve = self._make_request(client_host="10.0.0.5")
        key = resolve(req)
        assert key == "ip:10.0.0.5"

    def test_api_key_takes_priority_over_session(self):
        req, resolve = self._make_request(
            headers={"x-api-key": "key1"}, cookies={"session_id": "sess1"}
        )
        key = resolve(req)
        assert key == "key:key1"

    def test_unknown_ip_when_no_client(self):
        from agent.orchestrator.middleware import resolve_rate_limit_key

        req = MagicMock()
        req.headers = {}
        req.cookies = {}
        req.client = None
        key = resolve_rate_limit_key(req)
        assert key == "ip:unknown"


class TestPerSessionRateLimit:
    """Different sessions should have independent rate limits."""

    def test_api_key_users_have_separate_quotas(self):
        from agent.orchestrator.middleware import RateLimiter

        rl = RateLimiter(max_requests=2, window_seconds=60)

        # User A exhausts quota
        rl.is_allowed("key:user-a")
        rl.is_allowed("key:user-a")
        assert rl.is_allowed("key:user-a") is False

        # User B still has full quota
        assert rl.is_allowed("key:user-b") is True
        assert rl.is_allowed("key:user-b") is True
        assert rl.is_allowed("key:user-b") is False

    def test_session_and_ip_are_independent(self):
        from agent.orchestrator.middleware import RateLimiter

        rl = RateLimiter(max_requests=1, window_seconds=60)

        rl.is_allowed("session:s1")
        assert rl.is_allowed("session:s1") is False
        # Same IP but as a session-less request — separate bucket
        assert rl.is_allowed("ip:127.0.0.1") is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_uses_api_key(self):
        from agent.orchestrator.middleware import (
            check_rate_limit,
            rate_limiter,
        )

        # Save and replace with a tight limiter
        orig_max = rate_limiter.max_requests
        rate_limiter.max_requests = 2
        rate_limiter.reset()

        try:
            req = MagicMock()
            req.headers = {"x-api-key": "test-key"}
            req.cookies = {}
            req.client = MagicMock()
            req.client.host = "1.2.3.4"

            # Should succeed twice
            assert await check_rate_limit(req) is True
            assert await check_rate_limit(req) is True

            # Third should raise 429
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit(req)
            assert exc_info.value.status_code == 429
        finally:
            rate_limiter.max_requests = orig_max
            rate_limiter.reset()

    @pytest.mark.asyncio
    async def test_check_rate_limit_falls_back_to_ip(self):
        from agent.orchestrator.middleware import check_rate_limit, rate_limiter

        orig_max = rate_limiter.max_requests
        rate_limiter.max_requests = 1
        rate_limiter.reset()

        try:
            req = MagicMock()
            req.headers = {}
            req.cookies = {}
            req.client = MagicMock()
            req.client.host = "192.168.1.1"

            assert await check_rate_limit(req) is True

            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit(req)
            assert exc_info.value.status_code == 429
        finally:
            rate_limiter.max_requests = orig_max
            rate_limiter.reset()


class TestRateLimitEndpoint:
    """Integration tests for rate-limit-enforced endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from agent.orchestrator.server import app

        return TestClient(app)

    def test_health_is_not_rate_limited(self, client):
        """Health endpoint has no rate-limit dependency."""
        for _ in range(100):
            resp = client.get("/health")
            assert resp.status_code == 200
