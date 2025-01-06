#!/usr/bin/env python3

import argparse
import sys
import traceback
import time
import random
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

def get_random_user_agent():
    """Return a random User-Agent string."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    return random.choice(user_agents)

def search_with_retry(query, max_results=10, max_retries=3, initial_delay=2):
    """
    Perform search with retry mechanism.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        max_retries (int): Maximum number of retry attempts
        initial_delay (int): Initial delay between retries in seconds
    """
    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': get_random_user_agent()}
            
            print(f"DEBUG: Attempt {attempt + 1}/{max_retries} - Searching for query: {query}", 
                  file=sys.stderr)
            
            with DDGS(headers=headers) as ddgs:
                # Try API backend first, fallback to HTML if needed
                try:
                    results = list(ddgs.text(
                        query,
                        max_results=max_results,
                        backend='api'
                    ))
                except DuckDuckGoSearchException as api_error:
                    print(f"DEBUG: API backend failed, trying HTML backend: {str(api_error)}", 
                          file=sys.stderr)
                    # Add delay before trying HTML backend
                    time.sleep(1)
                    results = list(ddgs.text(
                        query,
                        max_results=max_results,
                        backend='html'
                    ))
                
                if not results:
                    print("DEBUG: No results found", file=sys.stderr)
                    return []
                
                print(f"DEBUG: Found {len(results)} results", file=sys.stderr)
                return results
                
        except Exception as e:
            print(f"ERROR: Attempt {attempt + 1} failed: {str(e)}", file=sys.stderr)
            if attempt < max_retries - 1:
                delay = initial_delay * (attempt + 1) + random.random() * 2
                print(f"DEBUG: Waiting {delay:.2f} seconds before retry...", file=sys.stderr)
                time.sleep(delay)
            else:
                print("ERROR: All retry attempts failed", file=sys.stderr)
                raise

def format_results(results):
    """Format and print search results."""
    for i, r in enumerate(results, 1):
        print(f"\n=== Result {i} ===")
        print(f"URL: {r.get('link', r.get('href', 'N/A'))}")
        print(f"Title: {r.get('title', 'N/A')}")
        print(f"Snippet: {r.get('snippet', r.get('body', 'N/A'))}")

def search(query, max_results=10):
    """
    Main search function that handles both API and HTML backends with retry mechanism.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
    """
    try:
        results = search_with_retry(query, max_results)
        if results:
            format_results(results)
            
    except Exception as e:
        print(f"ERROR: Search failed: {str(e)}", file=sys.stderr)
        print(f"ERROR type: {type(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Search using DuckDuckGo with fallback mechanisms")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=10,
                      help="Maximum number of results (default: 10)")
    
    args = parser.parse_args()
    search(args.query, args.max_results)

if __name__ == "__main__":
    main()
