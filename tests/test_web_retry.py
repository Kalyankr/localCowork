"""Tests for retry with backoff on web_search and fetch_webpage."""

from unittest.mock import MagicMock, patch

import requests

from agent.web import _MAX_RETRIES, _RETRY_BACKOFF, fetch_webpage, web_search


class TestWebSearchRetry:
    """web_search should retry on transient errors."""

    @patch("agent.web.DDGS")
    def test_succeeds_on_first_try(self, mock_ddgs_cls):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = [
            {"title": "Test", "href": "http://example.com", "body": "snippet"}
        ]
        mock_ddgs_cls.return_value = mock_ctx

        result = web_search("test query")
        assert "results" in result
        assert len(result["results"]) == 1

    @patch("agent.web.time.sleep")
    @patch("agent.web.DDGS")
    def test_retries_on_connection_error(self, mock_ddgs_cls, mock_sleep):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.side_effect = [
            requests.exceptions.ConnectionError("refused"),
            requests.exceptions.ConnectionError("refused"),
            [{"title": "OK", "href": "http://ok.com", "body": "ok"}],
        ]
        mock_ddgs_cls.return_value = mock_ctx

        result = web_search("test")
        assert "results" in result
        assert result["results"][0]["title"] == "OK"
        assert mock_sleep.call_count == 2

    @patch("agent.web.time.sleep")
    @patch("agent.web.DDGS")
    def test_gives_up_after_max_retries(self, mock_ddgs_cls, mock_sleep):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.side_effect = requests.exceptions.ConnectionError("down")
        mock_ddgs_cls.return_value = mock_ctx

        result = web_search("test")
        assert "error" in result
        assert "attempts" in result["error"]
        assert mock_sleep.call_count == _MAX_RETRIES

    @patch("agent.web.DDGS")
    def test_no_retry_on_non_transient_error(self, mock_ddgs_cls):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.side_effect = ValueError("bad query")
        mock_ddgs_cls.return_value = mock_ctx

        result = web_search("test")
        assert "error" in result
        assert "Search failed" in result["error"]

    @patch("agent.web.time.sleep")
    @patch("agent.web.DDGS")
    def test_backoff_doubling(self, mock_ddgs_cls, mock_sleep):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.side_effect = [
            requests.exceptions.Timeout("slow"),
            requests.exceptions.Timeout("slow"),
            [{"title": "OK", "href": "http://ok.com", "body": "ok"}],
        ]
        mock_ddgs_cls.return_value = mock_ctx

        web_search("test")
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls[0] == _RETRY_BACKOFF * 1  # 2^0
        assert calls[1] == _RETRY_BACKOFF * 2  # 2^1


class TestFetchWebpageRetry:
    """fetch_webpage should retry on transient errors."""

    @patch("agent.web.requests.get")
    def test_succeeds_on_first_try(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<html><head><title>T</title></head><body>Hello</body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_webpage("http://example.com")
        assert "content" in result
        assert result["title"] == "T"

    @patch("agent.web.time.sleep")
    @patch("agent.web.requests.get")
    def test_retries_on_timeout(self, mock_get, mock_sleep):
        good_resp = MagicMock()
        good_resp.text = "<html><head><title>OK</title></head><body>OK</body></html>"
        good_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [
            requests.exceptions.Timeout("slow"),
            good_resp,
        ]

        result = fetch_webpage("http://slow.com")
        assert "content" in result
        assert mock_sleep.call_count == 1

    @patch("agent.web.time.sleep")
    @patch("agent.web.requests.get")
    def test_gives_up_after_max_retries(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        result = fetch_webpage("http://down.com")
        assert "error" in result
        assert "attempts" in result["error"]

    @patch("agent.web.requests.get")
    def test_no_retry_on_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = mock_resp

        result = fetch_webpage("http://notfound.com")
        assert "error" in result
        assert mock_get.call_count == 1
