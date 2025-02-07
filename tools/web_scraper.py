#!/usr/bin/env python3

import asyncio
import sys
import os
from typing import List, Optional, Dict, Any, Tuple
from playwright.async_api import async_playwright
import html5lib
from multiprocessing import Pool
import time
from urllib.parse import urlparse
import aiohttp
from .common.logging_config import setup_logging
from .common.errors import ToolError, ValidationError, APIError
from .common.formatting import format_output, format_duration
from .common.cli import create_parser, get_log_config

logger = setup_logging(__name__)

class FetchError(APIError):
    """Custom exception for fetch failures"""
    def __init__(self, message: str, url: str, status_code: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        context = context or {}
        context["url"] = url
        if status_code is not None:
            context["status_code"] = status_code
        super().__init__(message, "fetch", context)
        self.url = url
        self.status_code = status_code

def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if URL is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except:
        return False

def validate_max_concurrent(max_concurrent: int) -> int:
    """
    Validate and normalize max_concurrent parameter.
    
    Args:
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        int: Normalized max_concurrent value
        
    Raises:
        ValidationError: If max_concurrent is invalid
    """
    if max_concurrent < 1:
        raise ValidationError("max_concurrent must be a positive integer", {
            "max_concurrent": max_concurrent
        })
    if max_concurrent > 20:  # Reasonable upper limit
        raise ValidationError("max_concurrent cannot exceed 20", {
            "max_concurrent": max_concurrent,
            "max_allowed": 20
        })
    return max_concurrent

async def fetch_page(url: str, session: Optional[aiohttp.ClientSession] = None, timeout: int = 30) -> str:
    """
    Asynchronously fetch a webpage's content.
    
    Args:
        url: URL to fetch
        session: Optional aiohttp session to reuse
        timeout: Request timeout in seconds
        
    Returns:
        str: Page content
        
    Raises:
        ValidationError: If URL is invalid
        FetchError: If the page cannot be fetched
    """
    if not validate_url(url):
        logger.error("Invalid URL format", extra={
            "context": {
                "url": url,
                "error": "Invalid URL format"
            }
        })
        raise ValidationError("Invalid URL format", {"url": url})
    
    logger.info("Fetching page", extra={
        "context": {
            "url": url,
            "timeout": timeout,
            "reuse_session": session is not None
        }
    })
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    start_time = time.time()
    try:
        if session is None:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        elapsed = time.time() - start_time
                        logger.info("Successfully fetched page", extra={
                            "context": {
                                "url": url,
                                "elapsed_seconds": elapsed,
                                "content_length": len(content)
                            }
                        })
                        return content
                    else:
                        logger.error("HTTP error", extra={
                            "context": {
                                "url": url,
                                "status_code": response.status,
                                "elapsed_seconds": time.time() - start_time
                            }
                        })
                        raise FetchError("HTTP error", url, response.status)
        else:
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.text()
                    elapsed = time.time() - start_time
                    logger.info("Successfully fetched page", extra={
                        "context": {
                            "url": url,
                            "elapsed_seconds": elapsed,
                            "content_length": len(content)
                        }
                    })
                    return content
                else:
                    logger.error("HTTP error", extra={
                        "context": {
                            "url": url,
                            "status_code": response.status,
                            "elapsed_seconds": time.time() - start_time
                        }
                    })
                    raise FetchError("HTTP error", url, response.status)
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error("Request timeout", extra={
            "context": {
                "url": url,
                "timeout": timeout,
                "elapsed_seconds": elapsed
            }
        })
        raise FetchError("Request timed out", url, None, {
            "timeout": timeout,
            "elapsed_seconds": elapsed
        })
    except aiohttp.ClientError as e:
        logger.error("Network error", extra={
            "context": {
                "url": url,
                "error": str(e),
                "elapsed_seconds": time.time() - start_time
            }
        })
        raise FetchError("Network error", url, None, {
            "error": str(e),
            "elapsed_seconds": time.time() - start_time
        })
    except Exception as e:
        logger.error("Unexpected error", extra={
            "context": {
                "url": url,
                "error": str(e),
                "elapsed_seconds": time.time() - start_time
            }
        })
        raise FetchError("Failed to fetch page", url, None, {
            "error": str(e),
            "elapsed_seconds": time.time() - start_time
        })

def parse_html(content: str, min_text_length: int = 50) -> str:
    """
    Parse HTML content and extract text with hyperlinks in markdown format.
    
    Args:
        content: HTML content to parse
        min_text_length: Minimum length for text blocks to be included
        
    Returns:
        str: Extracted text in markdown format
        
    Raises:
        ValidationError: If content is empty or invalid
    """
    if not content or not content.strip():
        raise ValidationError("Empty content")

    try:
        document = html5lib.parse(content)
        parsed_text = []
        seen_texts = set()

        def should_skip_element(elem) -> bool:
            if elem.tag in ['{http://www.w3.org/1999/xhtml}script', 
                          '{http://www.w3.org/1999/xhtml}style',
                          '{http://www.w3.org/1999/xhtml}noscript',
                          '{http://www.w3.org/1999/xhtml}iframe']:
                return True
            if not any(text.strip() for text in elem.itertext()):
                return True
            return False

        def process_element(elem, depth=0):
            if should_skip_element(elem):
                return

            if hasattr(elem, 'text') and elem.text:
                text = elem.text.strip()
                if text and text not in seen_texts:
                    if elem.tag == '{http://www.w3.org/1999/xhtml}a':
                        href = None
                        for attr, value in elem.items():
                            if attr.endswith('href'):
                                href = value
                                break
                        if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                            link_text = f"[{text}]({href})"
                            parsed_text.append("  " * depth + link_text)
                            seen_texts.add(text)
                    else:
                        if len(text) >= min_text_length:
                            parsed_text.append("  " * depth + text)
                            seen_texts.add(text)

            for child in elem:
                process_element(child, depth + 1)

            if hasattr(elem, 'tail') and elem.tail:
                tail = elem.tail.strip()
                if tail and tail not in seen_texts and len(tail) >= min_text_length:
                    parsed_text.append("  " * depth + tail)
                    seen_texts.add(tail)

        body = document.find('.//{http://www.w3.org/1999/xhtml}body')
        if body is not None:
            process_element(body)
        else:
            process_element(document)

        filtered_text = [
            line for line in parsed_text 
            if not any(pattern in line.lower() for pattern in [
                'var ', 'function()', '.js', '.css',
                'google-analytics', 'disqus', '{', '}',
                'cookie', 'privacy policy', 'terms of service'
            ])
        ]

        if not filtered_text:
            logger.warning("No meaningful content extracted from HTML")
            return ""

        return '\n'.join(filtered_text)

    except Exception as e:
        logger.error("Error parsing HTML", extra={
            "context": {
                "error": str(e)
            }
        })
        raise ValidationError("Failed to parse HTML", {
            "error": str(e)
        })

async def process_urls(urls: List[str], max_concurrent: int = 5, session: Optional[aiohttp.ClientSession] = None, timeout: int = 30) -> Dict[str, Any]:
    """
    Process multiple URLs concurrently.
    
    Args:
        urls: List of URLs to process
        max_concurrent: Maximum number of concurrent requests (1-20)
        session: Optional aiohttp session to reuse
        timeout: Request timeout in seconds
        
    Returns:
        Dict containing:
            - results: List of successfully parsed content
            - errors: Dict mapping failed URLs to their error messages
            
    Raises:
        ValidationError: If input parameters are invalid
    """
    # Validate inputs
    try:
        max_concurrent = validate_max_concurrent(max_concurrent)
    except ValidationError as e:
        logger.error("Invalid max_concurrent", extra={
            "context": e.context
        })
        raise
    
    # Filter out invalid URLs
    valid_urls = []
    for url in urls:
        if validate_url(url):
            valid_urls.append(url)
        else:
            logger.warning("Skipping invalid URL", extra={
                "context": {
                    "url": url,
                    "error": "Invalid URL format"
                }
            })
    
    if not valid_urls:
        logger.error("No valid URLs provided", extra={
            "context": {
                "total_urls": len(urls),
                "valid_urls": 0
            }
        })
        raise ValidationError("No valid URLs provided", {
            "total_urls": len(urls)
        })
    
    logger.info("Processing URLs", extra={
        "context": {
            "total_urls": len(urls),
            "valid_urls": len(valid_urls),
            "max_concurrent": max_concurrent,
            "timeout": timeout
        }
    })
    
    results = []
    errors = {}
    
    async def process_url(url: str, session: aiohttp.ClientSession) -> None:
        try:
            logger.debug("Starting URL processing", extra={
                "context": {
                    "url": url
                }
            })
            content = await fetch_page(url, session, timeout)
            parsed = parse_html(content)
            if parsed:
                results.append({
                    "url": url,
                    "content": parsed,
                    "timestamp": time.time()
                })
                logger.debug("Successfully processed URL", extra={
                    "context": {
                        "url": url,
                        "content_length": len(parsed)
                    }
                })
            else:
                logger.warning("No content extracted", extra={
                    "context": {
                        "url": url
                    }
                })
                errors[url] = "No meaningful content extracted"
        except Exception as e:
            logger.error("Failed to process URL", extra={
                "context": {
                    "url": url,
                    "error": str(e)
                }
            })
            errors[url] = str(e)

    if session is None:
        async with aiohttp.ClientSession() as session:
            tasks = [process_url(url, session) for url in valid_urls]
            # Process in batches to respect max_concurrent
            for i in range(0, len(tasks), max_concurrent):
                batch = tasks[i:i + max_concurrent]
                await asyncio.gather(*batch, return_exceptions=True)
    else:
        tasks = [process_url(url, session) for url in valid_urls]
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            await asyncio.gather(*batch, return_exceptions=True)

    logger.info("Completed URL processing", extra={
        "context": {
            "total_urls": len(valid_urls),
            "successful": len(results),
            "failed": len(errors)
        }
    })
    
    return {
        "results": results,
        "errors": errors
    }

def main():
    parser = create_parser('Fetch and process multiple URLs concurrently')
    parser.add_argument('urls', nargs='+', help='URLs to process')
    parser.add_argument('--max-concurrent', type=int, default=5, 
                       help='Maximum number of concurrent requests (1-20)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds')
    
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
        results = asyncio.run(process_urls(
            args.urls,
            args.max_concurrent,
            timeout=args.timeout
        ))
        elapsed = time.time() - start_time
        
        metadata = {
            "total_urls": len(args.urls),
            "successful": len(results["results"]),
            "failed": len(results["errors"]),
            "elapsed_time": format_duration(elapsed)
        }
        
        print(format_output(results, args.format, "Web Scraping Results", metadata))
        
        # Exit with error if any URLs failed
        if results["errors"]:
            sys.exit(1)
            
    except ValidationError as e:
        logger.error("Invalid input", extra={"context": e.context})
        sys.exit(1)
    except Exception as e:
        logger.error("Processing failed", extra={
            "context": {
                "error": str(e)
            }
        })
        sys.exit(1)

if __name__ == "__main__":
    main() 