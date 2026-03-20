"""Tests for the rate limiter middleware."""

import time

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
