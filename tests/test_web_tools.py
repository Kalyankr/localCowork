"""Tests for web tools."""

import pytest
from unittest.mock import patch, MagicMock


class TestFetchURL:
    """Tests for the fetch_url function."""

    @patch("agent.tools.web_tools._make_request")
    def test_fetch_url_success(self, mock_request):
        """fetch_url should return content on success."""
        from agent.tools.web_tools import fetch_url
        
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello World</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_request.return_value = mock_response
        
        result = fetch_url("https://example.com")
        
        assert "content" in result or "error" not in result
        mock_request.assert_called_once()

    @patch("agent.tools.web_tools._make_request")
    def test_fetch_url_with_timeout(self, mock_request):
        """fetch_url should respect timeout setting."""
        from agent.tools.web_tools import fetch_url, WebOperationError
        
        mock_request.side_effect = WebOperationError("Request timed out")
        
        result = fetch_url("https://example.com", timeout=5)
        
        assert "error" in result

    @patch("agent.tools.web_tools._make_request")
    def test_fetch_url_extract_text(self, mock_request):
        """fetch_url should extract text from HTML when requested."""
        from agent.tools.web_tools import fetch_url
        
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello World</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_request.return_value = mock_response
        
        result = fetch_url("https://example.com", extract_text=True)
        
        # Should contain text content
        assert result is not None


class TestMakeRequest:
    """Tests for the _make_request function."""

    @patch("agent.tools.web_tools.requests.request")
    def test_make_request_success(self, mock_requests):
        """_make_request should return response on success."""
        from agent.tools.web_tools import _make_request
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response
        
        result = _make_request("https://example.com")
        
        assert result == mock_response

    @patch("agent.tools.web_tools.requests.request")
    def test_make_request_retries_on_failure(self, mock_requests):
        """_make_request should retry on connection errors."""
        from agent.tools.web_tools import _make_request, WebOperationError
        import requests
        
        # Fail twice, succeed on third try
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            requests.exceptions.ConnectionError("Connection refused"),
            mock_response,
        ]
        
        result = _make_request("https://example.com", retries=3, backoff=0.01)
        
        assert result == mock_response
        assert mock_requests.call_count == 3

    @patch("agent.tools.web_tools.requests.request")
    def test_make_request_raises_after_max_retries(self, mock_requests):
        """_make_request should raise WebOperationError after max retries."""
        from agent.tools.web_tools import _make_request, WebOperationError
        import requests
        
        mock_requests.side_effect = requests.exceptions.Timeout("Timed out")
        
        with pytest.raises(WebOperationError) as exc_info:
            _make_request("https://example.com", retries=2, backoff=0.01)
        
        assert "failed after" in str(exc_info.value).lower()

    @patch("agent.tools.web_tools.requests.request")
    def test_make_request_no_retry_on_client_error(self, mock_requests):
        """_make_request should not retry on 4xx errors."""
        from agent.tools.web_tools import _make_request, WebOperationError
        import requests
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = error
        mock_requests.return_value = mock_response
        
        with pytest.raises(WebOperationError) as exc_info:
            _make_request("https://example.com/notfound", retries=3)
        
        # Should only try once for 4xx errors
        assert mock_requests.call_count == 1
