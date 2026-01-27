"""Web operations: fetch URLs, search the web, download files."""

import requests
import time
import logging
from urllib.parse import quote_plus
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.5
MAX_CONTENT_SIZE = 50000


class WebOperationError(Exception):
    """Exception raised for web operation errors."""

    pass


def _make_request(
    url: str,
    method: str = "GET",
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    headers: dict | None = None,
    stream: bool = False,
    **kwargs,
) -> requests.Response:
    """Make an HTTP request with retry logic and exponential backoff.

    Args:
        url: URL to request
        method: HTTP method
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        backoff: Backoff multiplier between retries
        headers: Optional custom headers
        stream: Whether to stream the response
        **kwargs: Additional arguments to pass to requests

    Returns:
        Response object

    Raises:
        WebOperationError: If all retries fail
    """
    default_headers = {"User-Agent": "Mozilla/5.0 (compatible; LocalCowork/1.0)"}
    if headers:
        default_headers.update(headers)

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.request(
                method,
                url,
                headers=default_headers,
                timeout=timeout,
                stream=stream,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(
                f"Request timed out (attempt {attempt + 1}/{retries}): {url}"
            )
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f"Connection error (attempt {attempt + 1}/{retries}): {url}")
        except requests.exceptions.HTTPError as e:
            # Don't retry client errors (4xx)
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise WebOperationError(f"HTTP {e.response.status_code}: {e}")
            last_error = e
            logger.warning(f"HTTP error (attempt {attempt + 1}/{retries}): {e}")
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")

        if attempt < retries - 1:
            sleep_time = backoff**attempt
            logger.debug(f"Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    raise WebOperationError(f"Request failed after {retries} attempts: {last_error}")


def fetch_url(
    url: str,
    extract_text: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict:
    """
    Fetch content from a URL with retry logic.

    Args:
        url: URL to fetch
        extract_text: If True, extract text from HTML
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Dict with 'content', 'status_code', 'content_type', 'url'
    """
    try:
        response = _make_request(url, timeout=timeout, retries=retries)
        content_type = response.headers.get("Content-Type", "")

        if extract_text and "text/html" in content_type:
            content = _extract_text_from_html(response.text)
        else:
            content = response.text

        return {
            "content": content[:MAX_CONTENT_SIZE],
            "status_code": response.status_code,
            "content_type": content_type,
            "url": url,
            "content_length": len(content),
        }

    except WebOperationError as e:
        return {"error": str(e), "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML content."""
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
            self.skip_tags = {"script", "style", "head", "meta", "link", "noscript"}
            self.current_tag = None
            self.tag_stack = []

        def handle_starttag(self, tag, attrs):
            self.tag_stack.append(tag)
            self.current_tag = tag

        def handle_endtag(self, tag):
            if self.tag_stack and self.tag_stack[-1] == tag:
                self.tag_stack.pop()
            self.current_tag = self.tag_stack[-1] if self.tag_stack else None

        def handle_data(self, data):
            # Check if any parent tag is in skip_tags
            if not any(t in self.skip_tags for t in self.tag_stack):
                text = data.strip()
                if text:
                    self.text.append(text)

    parser = TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass  # Return what we have
    return "\n".join(parser.text)


def search_web(
    query: str, num_results: int = 5, retries: int = DEFAULT_RETRIES
) -> list:
    """
    Search the web using DuckDuckGo HTML (no API key needed).

    Args:
        query: Search query
        num_results: Number of results to return
        retries: Number of retry attempts

    Returns:
        List of results with 'title', 'url', 'snippet'
    """
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    try:
        response = _make_request(search_url, timeout=15, retries=retries)

        # Parse results from HTML
        from html.parser import HTMLParser

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.in_result = False
                self.in_title = False
                self.in_snippet = False

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and attrs.get("class") == "result__a":
                    self.in_title = True
                    self.current["url"] = attrs.get("href", "")
                elif tag == "a" and "result__snippet" in attrs.get("class", ""):
                    self.in_snippet = True

            def handle_endtag(self, tag):
                if tag == "a" and self.in_title:
                    self.in_title = False
                elif tag == "a" and self.in_snippet:
                    self.in_snippet = False
                    if self.current.get("title") and self.current.get("url"):
                        self.results.append(self.current)
                    self.current = {}

            def handle_data(self, data):
                if self.in_title:
                    self.current["title"] = data.strip()
                elif self.in_snippet:
                    self.current["snippet"] = (
                        self.current.get("snippet", "") + data.strip()
                    )

        parser = DDGParser()
        parser.feed(response.text)

        return parser.results[:num_results]

    except WebOperationError as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": str(e)}]


def download_file(
    url: str,
    dest: str,
    timeout: int = 60,
    retries: int = DEFAULT_RETRIES,
    overwrite: bool = False,
) -> dict:
    """
    Download a file from URL to destination path with retry logic.

    Args:
        url: URL to download
        dest: Destination file path
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        overwrite: If False, skip download if file exists

    Returns:
        Dict with download info or error
    """
    dest_path = Path(dest).expanduser()

    # Check if file exists
    if dest_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "message": f"File already exists: {dest_path}",
            "path": str(dest_path),
        }

    try:
        response = _make_request(url, timeout=timeout, retries=retries, stream=True)

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = 0

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        return {
            "status": "success",
            "message": f"Downloaded {url} → {dest_path}",
            "path": str(dest_path),
            "size": downloaded,
            "size_human": _format_size(downloaded),
        }

    except WebOperationError as e:
        return {"status": "error", "error": str(e), "url": url}
    except Exception as e:
        return {"status": "error", "error": str(e), "url": url}


def check_url(url: str, timeout: int = 10) -> dict:
    """
    Check if a URL is accessible without downloading full content.

    Args:
        url: URL to check
        timeout: Request timeout in seconds

    Returns:
        Dict with status info
    """
    try:
        response = requests.head(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LocalCowork/1.0)"},
            timeout=timeout,
            allow_redirects=True,
        )

        return {
            "accessible": response.status_code < 400,
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "content_length": response.headers.get("Content-Length"),
            "url": url,
            "final_url": response.url,
        }
    except Exception as e:
        return {
            "accessible": False,
            "error": str(e),
            "url": url,
        }


def _format_size(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def dispatch(op: str, **kwargs) -> dict | list | str:
    """Dispatch web operations.

    Supported operations:
        - fetch: Fetch URL content
        - search: Search web using DuckDuckGo
        - download: Download file from URL
        - check: Check if URL is accessible
    """
    if op == "fetch":
        return fetch_url(
            kwargs["url"],
            kwargs.get("extract_text", True),
            kwargs.get("timeout", DEFAULT_TIMEOUT),
            kwargs.get("retries", DEFAULT_RETRIES),
        )
    if op == "search":
        return search_web(
            kwargs["query"],
            kwargs.get("num_results", 5),
            kwargs.get("retries", DEFAULT_RETRIES),
        )
    if op == "download":
        return download_file(
            kwargs["url"],
            kwargs["dest"],
            kwargs.get("timeout", 60),
            kwargs.get("retries", DEFAULT_RETRIES),
            kwargs.get("overwrite", False),
        )
    if op == "check":
        return check_url(kwargs["url"], kwargs.get("timeout", 10))
    raise ValueError(f"Unsupported web op: {op}")
