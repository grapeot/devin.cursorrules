#!/usr/bin/env /workspace/tmp_windsurf/venv/bin/python3

import asyncio
import argparse
import sys
import os
from typing import List, Optional
from playwright.async_api import async_playwright
import html5lib
from multiprocessing import Pool
import time
from urllib.parse import urlparse
import logging
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

async def fetch_page(url: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[str]:
    """Asynchronously fetch a webpage's content."""
    if session is None:
        async with aiohttp.ClientSession() as session:
            try:
                logger.info(f"Fetching {url}")
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Successfully fetched {url}")
                        return content
                    else:
                        logger.error(f"Error fetching {url}: HTTP {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                return None
    else:
        try:
            logger.info(f"Fetching {url}")
            response = await session.get(url)
            if response.status == 200:
                content = await response.text()
                logger.info(f"Successfully fetched {url}")
                return content
            else:
                logger.error(f"Error fetching {url}: HTTP {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

def parse_html(html_content: Optional[str]) -> str:
    """Parse HTML content and extract text with hyperlinks in markdown format."""
    if not html_content:
        return ""
    
    try:
        document = html5lib.parse(html_content)
        result = []
        seen_texts = set()  # To avoid duplicates
        
        def should_skip_element(elem) -> bool:
            """Check if the element should be skipped."""
            # Skip script and style tags
            if elem.tag in ['{http://www.w3.org/1999/xhtml}script', 
                          '{http://www.w3.org/1999/xhtml}style']:
                return True
            # Skip empty elements or elements with only whitespace
            if not any(text.strip() for text in elem.itertext()):
                return True
            return False
        
        def process_element(elem, depth=0):
            """Process an element and its children recursively."""
            if should_skip_element(elem):
                return
            
            # Handle text content
            if hasattr(elem, 'text') and elem.text:
                text = elem.text.strip()
                if text and text not in seen_texts:
                    # Check if this is an anchor tag
                    if elem.tag == '{http://www.w3.org/1999/xhtml}a':
                        href = None
                        for attr, value in elem.items():
                            if attr.endswith('href'):
                                href = value
                                break
                        if href and not href.startswith(('#', 'javascript:')):
                            # Format as markdown link
                            link_text = f"[{text}]({href})"
                            result.append("  " * depth + link_text)
                            seen_texts.add(text)
                    else:
                        result.append("  " * depth + text)
                        seen_texts.add(text)
            
            # Process children
            for child in elem:
                process_element(child, depth + 1)
            
            # Handle tail text
            if hasattr(elem, 'tail') and elem.tail:
                tail = elem.tail.strip()
                if tail and tail not in seen_texts:
                    result.append("  " * depth + tail)
                    seen_texts.add(tail)
        
        # Start processing from the body tag
        body = document.find('.//{http://www.w3.org/1999/xhtml}body')
        if body is not None:
            process_element(body)
        else:
            # Fallback to processing the entire document
            process_element(document)
        
        # Filter out common unwanted patterns
        filtered_result = []
        for line in result:
            # Skip lines that are likely to be noise
            if any(pattern in line.lower() for pattern in [
                'var ', 
                'function()', 
                '.js',
                '.css',
                'google-analytics',
                'disqus',
                '{',
                '}'
            ]):
                continue
            filtered_result.append(line)
        
        return '\n'.join(filtered_result)
    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
        return ""

async def process_urls(urls: List[str], max_concurrent: int = 5, session: Optional[aiohttp.ClientSession] = None) -> List[str]:
    """Process multiple URLs concurrently."""
    if session is None:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_page(url, session) for url in urls]
            html_contents = await asyncio.gather(*tasks)
    else:
        tasks = [fetch_page(url, session) for url in urls]
        html_contents = await asyncio.gather(*tasks)
    
    # Parse HTML contents in parallel
    with Pool() as pool:
        results = pool.map(parse_html, html_contents)
    
    return results

def validate_url(url: str) -> bool:
    """Validate if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def main():
    """Main function to process URLs from command line."""
    parser = argparse.ArgumentParser(description='Fetch and process multiple URLs concurrently.')
    parser.add_argument('urls', nargs='+', help='URLs to process')
    parser.add_argument('--max-concurrent', type=int, default=5, help='Maximum number of concurrent requests')
    args = parser.parse_args()
    
    # Validate URLs
    valid_urls = [url for url in args.urls if validate_url(url)]
    if not valid_urls:
        logger.error("No valid URLs provided")
        sys.exit(1)
    
    # Process URLs
    results = asyncio.run(process_urls(valid_urls, args.max_concurrent))
    
    # Print results
    for url, content in zip(valid_urls, results):
        print(f"\n=== Content from {url} ===\n")
        print(content)

if __name__ == '__main__':
    main() 