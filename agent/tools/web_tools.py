"""Web operations: fetch URLs, search the web."""

import requests
from urllib.parse import quote_plus
from typing import Optional


def fetch_url(url: str, extract_text: bool = True) -> dict:
    """
    Fetch content from a URL.
    Returns dict with 'content', 'status_code', 'content_type'.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LocalCowork/1.0)"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "")
        
        if extract_text and "text/html" in content_type:
            # Simple HTML to text extraction
            from html.parser import HTMLParser
            
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
                    self.current_tag = None
                    
                def handle_starttag(self, tag, attrs):
                    self.current_tag = tag
                    
                def handle_endtag(self, tag):
                    self.current_tag = None
                    
                def handle_data(self, data):
                    if self.current_tag not in self.skip_tags:
                        text = data.strip()
                        if text:
                            self.text.append(text)
            
            parser = TextExtractor()
            parser.feed(response.text)
            content = "\n".join(parser.text)
        else:
            content = response.text
        
        return {
            "content": content[:50000],  # Limit content size
            "status_code": response.status_code,
            "content_type": content_type,
            "url": url,
        }
        
    except requests.RequestException as e:
        return {
            "error": str(e),
            "url": url,
        }


def search_web(query: str, num_results: int = 5) -> list:
    """
    Search the web using DuckDuckGo HTML (no API key needed).
    Returns list of results with 'title', 'url', 'snippet'.
    """
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LocalCowork/1.0)"
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        
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
                    self.current["snippet"] = self.current.get("snippet", "") + data.strip()
        
        parser = DDGParser()
        parser.feed(response.text)
        
        return parser.results[:num_results]
        
    except Exception as e:
        return [{"error": str(e)}]


def download_file(url: str, dest: str) -> str:
    """
    Download a file from URL to destination path.
    """
    from pathlib import Path
    
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LocalCowork/1.0)"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()
        
        dest_path = Path(dest).expanduser()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return f"Downloaded {url} → {dest_path}"
        
    except Exception as e:
        return f"Error downloading: {e}"


def dispatch(op: str, **kwargs):
    """Dispatch web operations."""
    if op == "fetch":
        return fetch_url(kwargs["url"], kwargs.get("extract_text", True))
    if op == "search":
        return search_web(kwargs["query"], kwargs.get("num_results", 5))
    if op == "download":
        return download_file(kwargs["url"], kwargs["dest"])
    raise ValueError(f"Unsupported web op: {op}")
