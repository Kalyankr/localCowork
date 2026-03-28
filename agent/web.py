"""Web search and browsing tools for LocalCowork.

This module provides web search and content fetching capabilities
without requiring browser automation.
"""

import time
from typing import Any

import requests
import structlog
from bs4 import BeautifulSoup
from ddgs import DDGS

from agent.tokens import truncate_to_tokens

logger = structlog.get_logger(__name__)

# Request timeout in seconds
REQUEST_TIMEOUT = 15

# Retry settings
_MAX_RETRIES = 2
_RETRY_BACKOFF = 1.0  # seconds, doubled each retry

# Transient exception types worth retrying
_TRANSIENT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionResetError,
    TimeoutError,
    OSError,
)

# User agent to avoid blocks
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        Dict with 'results' list or 'error' string
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            logger.debug("web_search", query=query, attempt=attempt)

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return {"results": [], "message": "No results found"}

            # Format results
            formatted = []
            for r in results:
                formatted.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("link", "")),
                        "snippet": r.get("body", r.get("snippet", "")),
                    }
                )

            return {"results": formatted}

        except _TRANSIENT_ERRORS as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BACKOFF * (2**attempt)
                logger.warning(
                    "web_search_retry", attempt=attempt, delay=delay, error=str(e)
                )
                time.sleep(delay)
                continue
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"error": f"Search failed: {str(e)}"}

    return {"error": f"Search failed after {_MAX_RETRIES + 1} attempts: {last_error}"}


def fetch_webpage(url: str, extract_text: bool = True) -> dict[str, Any]:
    """
    Fetch content from a webpage.

    Args:
        url: The URL to fetch
        extract_text: If True, extract readable text. If False, return raw HTML.

    Returns:
        Dict with 'content', 'title', 'url' or 'error'
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            logger.debug("fetch_webpage", url=url, attempt=attempt)

            headers = {"User-Agent": USER_AGENT}
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Get title
            title = soup.title.string if soup.title else ""

            if extract_text:
                # Remove script and style elements
                for element in soup(
                    ["script", "style", "nav", "footer", "header", "aside"]
                ):
                    element.decompose()

                # Get text content
                text = soup.get_text(separator="\n", strip=True)

                # Clean up excessive whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                content = "\n".join(lines)

                # Truncate if too long (keep first ~2000 tokens for context window)
                content = truncate_to_tokens(content, 2000)
            else:
                content = response.text

            return {
                "title": title,
                "url": url,
                "content": content,
            }

        except requests.exceptions.HTTPError as e:
            # Don't retry HTTP errors (4xx, 5xx) — they're not transient
            return {"error": f"Failed to fetch {url}: {str(e)}"}
        except _TRANSIENT_ERRORS as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BACKOFF * (2**attempt)
                logger.warning(
                    "fetch_retry", attempt=attempt, delay=delay, error=str(e)
                )
                time.sleep(delay)
                continue
            return {
                "error": f"Failed to fetch {url} after {_MAX_RETRIES + 1} attempts: {e}"
            }
        except Exception as e:
            logger.error(f"Webpage fetch error: {e}")
            return {"error": f"Error processing {url}: {str(e)}"}

    return {
        "error": f"Failed to fetch {url} after {_MAX_RETRIES + 1} attempts: {last_error}"
    }


def search_and_summarize(query: str, max_results: int = 3) -> dict[str, Any]:
    """
    Search the web and fetch content from top results.

    Args:
        query: Search query
        max_results: Number of pages to fetch (default 3)

    Returns:
        Dict with search results and their content
    """
    # First search
    search_result = web_search(query, max_results=max_results)

    if "error" in search_result:
        return search_result

    results = search_result.get("results", [])
    if not results:
        return {"results": [], "message": "No results found"}

    # Fetch content from each result
    enriched = []
    for r in results:
        url = r.get("url", "")
        if url:
            page = fetch_webpage(url)
            enriched.append(
                {
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": r.get("snippet", ""),
                    "content": page.get(
                        "content", page.get("error", "Could not fetch")
                    ),
                }
            )

    return {"results": enriched}
