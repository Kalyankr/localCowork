"""Tests for web search and fetch tools (agent.web)."""

from unittest.mock import MagicMock, patch

# =============================================================================
# web_search tests
# =============================================================================


class TestWebSearch:
    """Tests for the web_search function."""

    @patch("agent.web.DDGS")
    def test_returns_formatted_results(self, mock_ddgs_cls):
        from agent.web import web_search

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = [
            {"title": "Result 1", "href": "https://example.com", "body": "snippet 1"},
            {"title": "Result 2", "href": "https://example.org", "body": "snippet 2"},
        ]
        mock_ddgs_cls.return_value = mock_ctx

        result = web_search("python asyncio", max_results=2)

        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Result 1"
        assert result["results"][0]["url"] == "https://example.com"

    @patch("agent.web.DDGS")
    def test_empty_results(self, mock_ddgs_cls):
        from agent.web import web_search

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = []
        mock_ddgs_cls.return_value = mock_ctx

        result = web_search("very obscure query")
        assert result["results"] == []
        assert "message" in result

    @patch("agent.web.DDGS")
    def test_handles_search_exception(self, mock_ddgs_cls):
        from agent.web import web_search

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.side_effect = RuntimeError("network down")
        mock_ddgs_cls.return_value = mock_ctx

        result = web_search("test")
        assert "error" in result

    @patch("agent.web.DDGS")
    def test_max_results_is_passed_through(self, mock_ddgs_cls):
        from agent.web import web_search

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = []
        mock_ddgs_cls.return_value = mock_ctx

        web_search("test", max_results=7)
        mock_ctx.text.assert_called_once_with("test", max_results=7)


# =============================================================================
# fetch_webpage tests
# =============================================================================


class TestFetchWebpage:
    """Tests for the fetch_webpage function."""

    @patch("agent.web.requests.get")
    def test_returns_extracted_text(self, mock_get):
        from agent.web import fetch_webpage

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
        <html>
        <head><title>Test Page</title></head>
        <body><p>Hello world</p></body>
        </html>
        """
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_webpage("https://example.com")

        assert result["title"] == "Test Page"
        assert "Hello world" in result["content"]
        assert result["url"] == "https://example.com"

    @patch("agent.web.requests.get")
    def test_strips_script_and_style_tags(self, mock_get):
        from agent.web import fetch_webpage

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
        <html>
        <head><title>T</title><style>body{color:red}</style></head>
        <body>
            <script>alert('xss')</script>
            <p>Visible content</p>
            <nav>Nav stuff</nav>
        </body>
        </html>
        """
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_webpage("https://example.com")
        assert "alert" not in result["content"]
        assert "color:red" not in result["content"]
        assert "Nav stuff" not in result["content"]
        assert "Visible content" in result["content"]

    @patch("agent.web.requests.get")
    def test_returns_raw_html_when_extract_text_false(self, mock_get):
        from agent.web import fetch_webpage

        html = "<html><body><p>raw</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_webpage("https://example.com", extract_text=False)
        assert "<p>raw</p>" in result["content"]

    @patch("agent.web.requests.get")
    def test_handles_timeout(self, mock_get):
        import requests

        from agent.web import fetch_webpage

        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        result = fetch_webpage("https://slow.example.com")
        assert "error" in result
        assert "timed out" in result["error"].lower()

    @patch("agent.web.requests.get")
    def test_handles_connection_error(self, mock_get):
        import requests

        from agent.web import fetch_webpage

        mock_get.side_effect = requests.exceptions.ConnectionError("refused")
        result = fetch_webpage("https://down.example.com")
        assert "error" in result


# =============================================================================
# search_and_summarize tests
# =============================================================================


class TestSearchAndSummarize:
    """Tests for the combined search_and_summarize function."""

    @patch("agent.web.fetch_webpage")
    @patch("agent.web.web_search")
    def test_enriches_search_results_with_content(self, mock_search, mock_fetch):
        from agent.web import search_and_summarize

        mock_search.return_value = {
            "results": [
                {"title": "Page", "url": "https://example.com", "snippet": "snip"}
            ]
        }
        mock_fetch.return_value = {"content": "Full page content"}

        result = search_and_summarize("query")
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Full page content"

    @patch("agent.web.web_search")
    def test_returns_error_when_search_fails(self, mock_search):
        from agent.web import search_and_summarize

        mock_search.return_value = {"error": "search failed"}
        result = search_and_summarize("query")
        assert "error" in result

    @patch("agent.web.web_search")
    def test_returns_empty_when_no_results(self, mock_search):
        from agent.web import search_and_summarize

        mock_search.return_value = {"results": []}
        result = search_and_summarize("query")
        assert result["results"] == []
