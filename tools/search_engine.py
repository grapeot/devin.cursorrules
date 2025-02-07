#!/usr/bin/env python3

import sys
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from googlesearch import search as google_search
from .common.logging_config import setup_logging
from .common.errors import ToolError, ValidationError, APIError
from .common.formatting import format_output, format_duration
from .common.cli import create_parser, get_log_config

logger = setup_logging(__name__)

class SearchError(APIError):
    """Custom exception for search failures"""
    def __init__(self, message: str, engine: str, query: str, context: Optional[Dict[str, Any]] = None):
        context = context or {}
        context["query"] = query
        super().__init__(message, engine, context)
        self.query = query

class SearchEngine(ABC):
    """Abstract base class for search engines"""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10, max_retries: int = 3, fetch_snippets: bool = True) -> List[Dict[str, Any]]:
        """
        Execute a search query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            max_retries: Maximum number of retry attempts
            fetch_snippets: Whether to fetch page content for snippets
            
        Returns:
            List of dicts containing search results
            
        Raises:
            SearchError: If search operation fails
        """
        pass

def fetch_page_snippet(url: str, query: str, max_retries: int = 3) -> tuple[str, str]:
    """
    Fetch a webpage and extract its title and a snippet of content.
    
    Args:
        url: URL to fetch
        query: The search query that led to this URL
        max_retries: Maximum number of retry attempts
        
    Returns:
        tuple: (title, snippet)
        
    Raises:
        SearchError: If page cannot be fetched or parsed
    """
    start_time = time.time()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching snippet from {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title = soup.title.string if soup.title else url
            title = title.strip()
            
            # Get snippet from meta description or first paragraph
            snippet = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                snippet = meta_desc['content'].strip()
                logger.debug(f"Using meta description for {url} ({len(snippet)} chars)")
            else:
                # Try to get first few paragraphs
                paragraphs = soup.find_all('p')
                text_chunks = []
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and len(text) > 50:  # Skip short paragraphs
                        text_chunks.append(text)
                        if len(' '.join(text_chunks)) > 200:  # Get enough text
                            break
                if text_chunks:
                    snippet = ' '.join(text_chunks)[:300] + '...'  # Limit length
                    logger.debug(f"Using paragraph content for {url} ({len(snippet)} chars)")
                else:
                    logger.warning(f"No suitable content found for snippet in {url}")
            
            elapsed = time.time() - start_time
            logger.info(f"Successfully fetched snippet from {url} in {elapsed:.2f}s")
            return title, snippet
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching {url} (attempt {attempt + 1}/{max_retries}): {e.response.status_code}")
            if attempt < max_retries - 1:
                logger.debug("Waiting 1 second before retry...")
                time.sleep(1)
            else:
                raise SearchError(f"HTTP error {e.response.status_code}", "fetch", query, {
                    "url": url,
                    "status_code": e.response.status_code
                })
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {url} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                logger.debug("Waiting 1 second before retry...")
                time.sleep(1)
            else:
                raise SearchError("Request timed out", "fetch", query, {
                    "url": url,
                    "timeout": timeout
                })
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching {url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.debug("Waiting 1 second before retry...")
                time.sleep(1)
            else:
                raise SearchError("Network error", "fetch", query, {
                    "url": url,
                    "error": str(e)
                })
        except Exception as e:
            logger.error(f"Unexpected error fetching {url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.debug("Waiting 1 second before retry...")
                time.sleep(1)
            else:
                raise SearchError("Failed to fetch page", "fetch", query, {
                    "url": url,
                    "error": str(e)
                })

class DuckDuckGoEngine(SearchEngine):
    """DuckDuckGo search implementation"""
    
    def search(self, query: str, max_results: int = 10, max_retries: int = 3, fetch_snippets: bool = True) -> List[Dict[str, Any]]:
        start_time = time.time()
        logger.info(f"Starting DuckDuckGo search for: {query}")
        logger.debug(f"Search parameters: max_results={max_results}, max_retries={max_retries}")
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Search attempt {attempt + 1}/{max_retries}")
                
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                
                if not results:
                    logger.debug("No results found")
                    return []
                
                result_count = len(results)
                logger.debug(f"Found {result_count} results")
                
                # Normalize result format
                formatted_results = []
                for i, r in enumerate(results, 1):
                    formatted_result = {
                        "url": r.get("href", ""),
                        "title": r.get("title", ""),
                        "snippet": r.get("body", "")
                    }
                    
                    # Validate and clean up result
                    if not formatted_result["url"]:
                        logger.warning(f"Result {i}/{result_count} missing URL, skipping")
                        continue
                    
                    if not formatted_result["title"]:
                        logger.debug(f"Result {i}/{result_count} missing title, using URL")
                        formatted_result["title"] = formatted_result["url"]
                    
                    if not formatted_result["snippet"]:
                        logger.debug(f"Result {i}/{result_count} missing snippet")
                        formatted_result["snippet"] = ""
                    
                    logger.debug(f"Result {i}/{result_count}: {formatted_result['url']}")
                    formatted_results.append(formatted_result)
                
                elapsed = time.time() - start_time
                logger.info(f"Search completed in {elapsed:.2f}s with {len(formatted_results)} valid results")
                return formatted_results
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise SearchError("DuckDuckGo search failed", "duckduckgo", query, {
                        "error": str(e),
                        "attempts": max_retries
                    })
                time.sleep(1)
                continue

class GoogleEngine(SearchEngine):
    """Google search implementation"""
    
    def search(self, query: str, max_results: int = 10, max_retries: int = 3, fetch_snippets: bool = True) -> List[Dict[str, Any]]:
        start_time = time.time()
        logger.info(f"Starting Google search for: {query}")
        logger.debug(f"Search parameters: max_results={max_results}, max_retries={max_retries}, fetch_snippets={fetch_snippets}")
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Search attempt {attempt + 1}/{max_retries}")
                
                results = []
                urls = list(google_search(query, num_results=max_results))
                
                if not urls:
                    logger.info("No results found")
                    return []
                
                logger.debug(f"Found {len(urls)} URLs")
                
                for i, url in enumerate(urls, 1):
                    try:
                        if fetch_snippets:
                            logger.debug(f"Fetching content for result {i}/{len(urls)}: {url}")
                            title, snippet = fetch_page_snippet(url, query, max_retries)
                        else:
                            logger.debug(f"Skipping content fetch for result {i}/{len(urls)}: {url}")
                            title, snippet = url, ""
                            
                        result = {
                            "url": url,
                            "title": title,
                            "snippet": snippet
                        }
                        logger.debug(f"Result {i}/{len(urls)}: {url}")
                        results.append(result)
                        
                    except SearchError as e:
                        logger.warning(f"Failed to fetch snippet for {url}: {str(e)}")
                        # Include the result anyway, just without a snippet
                        results.append({
                            "url": url,
                            "title": url,
                            "snippet": ""
                        })
                
                if not results:
                    logger.info("No valid results found")
                    return []
                
                elapsed = time.time() - start_time
                logger.info(f"Search completed in {elapsed:.2f}s with {len(results)} results")
                return results
                
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.debug("Waiting 1 second before retry...")
                    time.sleep(1)
                else:
                    raise SearchError("Google search failed", "google", query, {
                        "error": str(e),
                        "attempts": max_retries
                    })

def get_search_engine(engine: str = "duckduckgo") -> SearchEngine:
    """
    Get a search engine instance by name.
    
    Args:
        engine: Name of the search engine ('duckduckgo' or 'google')
        
    Returns:
        SearchEngine instance
        
    Raises:
        ValidationError: If engine name is invalid
    """
    engines = {
        "duckduckgo": DuckDuckGoEngine,
        "google": GoogleEngine
    }
    
    engine_class = engines.get(engine.lower())
    if not engine_class:
        raise ValidationError("Invalid search engine", {
            "engine": engine,
            "valid_options": list(engines.keys())
        })
    
    return engine_class()

def validate_query(query: str) -> str:
    """
    Validate and normalize search query.
    
    Args:
        query: Search query string
        
    Returns:
        str: Normalized query
        
    Raises:
        ValidationError: If query is invalid
    """
    if not query or not query.strip():
        raise ValidationError("Search query cannot be empty")
    return query.strip()

def validate_max_results(max_results: int) -> int:
    """
    Validate and normalize max_results parameter.
    
    Args:
        max_results: Maximum number of results to return
        
    Returns:
        int: Normalized max_results value
        
    Raises:
        ValidationError: If max_results is invalid
    """
    if max_results < 1:
        raise ValidationError("max_results must be a positive integer", {
            "max_results": max_results
        })
    if max_results > 100:  # Reasonable upper limit
        raise ValidationError("max_results cannot exceed 100", {
            "max_results": max_results,
            "max_allowed": 100
        })
    return max_results

def search(query: str, max_results: int = 10, max_retries: int = 3, engine: str = "duckduckgo", fetch_snippets: bool = True) -> List[Dict[str, Any]]:
    """
    Search using the specified engine and return results with URLs and text snippets.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return (1-100)
        max_retries: Maximum number of retry attempts
        engine: Search engine to use ('duckduckgo' or 'google')
        fetch_snippets: Whether to fetch page content for snippets (Google only)
        
    Returns:
        List of dicts containing search results with keys:
            - url: Result URL
            - title: Result title
            - snippet: Text snippet
            
    Raises:
        ValidationError: If input parameters are invalid
        SearchError: If search operation fails
    """
    # Validate inputs
    try:
        query = validate_query(query)
        max_results = validate_max_results(max_results)
    except ValidationError as e:
        logger.error("Invalid input parameters", extra={
            "context": e.context
        })
        raise
    
    logger.info(f"Searching for: {query}", extra={
        "context": {
            "engine": engine,
            "max_results": max_results,
            "max_retries": max_retries,
            "fetch_snippets": fetch_snippets
        }
    })
    
    start_time = time.time()
    try:
        # Get search engine instance
        search_engine = get_search_engine(engine)
        
        # Perform search
        results = search_engine.search(query, max_results, max_retries, fetch_snippets)
        
        # Log results
        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.2f}s", extra={
            "context": {
                "engine": engine,
                "result_count": len(results),
                "elapsed_seconds": elapsed
            }
        })
        
        return results
        
    except ValidationError as e:
        logger.error("Invalid search engine", extra={
            "context": e.context
        })
        raise
    except SearchError as e:
        logger.error("Search failed", extra={
            "context": e.context
        })
        raise
    except Exception as e:
        logger.error("Unexpected error during search", extra={
            "context": {
                "engine": engine,
                "query": query,
                "error": str(e)
            }
        })
        raise SearchError("Search failed", engine, query, {
            "error": str(e)
        })

def main():
    parser = create_parser("Search using various search engines")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--engine", choices=["duckduckgo", "google"], default="duckduckgo",
                      help="Search engine to use")
    parser.add_argument("--max-results", type=int, default=10,
                      help="Maximum number of results")
    parser.add_argument("--max-retries", type=int, default=3,
                      help="Maximum number of retry attempts")
    parser.add_argument("--no-fetch-snippets", action="store_true",
                      help="Don't fetch page content for snippets (Google only)")
    
    args = parser.parse_args()

    # Configure logging
    log_config = get_log_config(args)
    logger = setup_logging(__name__, **log_config)
    logger.debug("Debug logging enabled", extra={
        "context": {
            "log_level": log_config["level"].value,
            "log_format": log_config["format_type"].value
        }
    })
    
    try:
        start_time = time.time()
        results = search(
            args.query,
            args.max_results,
            args.max_retries,
            args.engine,
            not args.no_fetch_snippets
        )
        elapsed = time.time() - start_time
        
        metadata = {
            "engine": args.engine,
            "query": args.query,
            "elapsed_time": format_duration(elapsed),
            "result_count": len(results)
        }
        
        print(format_output(results, args.format, "Search Results", metadata))
        
        # Exit with error if no results found
        if not results:
            sys.exit(1)
            
    except ValidationError as e:
        logger.error("Invalid input", extra={"context": e.context})
        sys.exit(1)
    except SearchError as e:
        logger.error(str(e), extra={"context": e.context})
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", extra={
            "context": {
                "error": str(e)
            }
        })
        sys.exit(1)

if __name__ == "__main__":
    main()
