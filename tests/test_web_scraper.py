#!/usr/bin/env python3

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import aiohttp
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import html5lib

from tools.web_scraper import (
    fetch_page,
    validate_url,
    validate_max_concurrent,
    parse_html,
    process_urls,
    FetchError
)
from tools.common.errors import ValidationError
from tools.common.formatting import format_output

class AsyncContextManagerMock:
    def __init__(self, response):
        self.response = response
    
    async def __aenter__(self):
        return self.response
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

def test_validate_url():
    """Test URL validation"""
    # Test valid URLs
    assert validate_url('https://example.com') == True
    assert validate_url('http://example.com/path?query=1') == True
    assert validate_url('https://sub.example.com:8080/path') == True
    
    # Test invalid URLs
    assert validate_url('not-a-url') == False
    assert validate_url('http://') == False
    assert validate_url('https://') == False
    assert validate_url('') == False
    assert validate_url('javascript:alert(1)') == False
    assert validate_url('ftp://example.com') == False

def test_validate_max_concurrent():
    """Test concurrent request validation"""
    # Test valid values
    assert validate_max_concurrent(1) == 1
    assert validate_max_concurrent(10) == 10
    assert validate_max_concurrent(20) == 20
    
    # Test invalid values
    with pytest.raises(ValidationError) as exc_info:
        validate_max_concurrent(0)
    assert "must be a positive integer" in str(exc_info.value)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_max_concurrent(-1)
    assert "must be a positive integer" in str(exc_info.value)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_max_concurrent(21)
    assert "cannot exceed 20" in str(exc_info.value)

def test_parse_html():
    """Test HTML parsing"""
    # Test with empty input
    with pytest.raises(ValidationError):
        parse_html("")
    
    with pytest.raises(ValidationError):
        parse_html("   ")
    
    # Test with simple HTML
    html = """
    <html>
        <body>
            <h1>Title</h1>
            <p>Short text</p>
            <p>This is a longer paragraph that should be included in the output because it exceeds the minimum length requirement</p>
            <a href="https://example.com">Link text that is long enough to be included in the output</a>
            <script>var x = 1;</script>
            <style>.css { color: red; }</style>
            <noscript>JavaScript is disabled</noscript>
            <iframe src="ad.html">Advertisement</iframe>
        </body>
    </html>
    """
    result = parse_html(html)
    assert "This is a longer paragraph" in result
    assert "[Link text that is long enough to be included in the output](https://example.com)" in result
    assert "Short text" not in result  # Too short
    assert "var x = 1" not in result  # In script tag
    assert ".css" not in result  # In style tag
    assert "JavaScript is disabled" not in result  # In noscript tag
    assert "Advertisement" not in result  # In iframe tag
    
    # Test with complex HTML
    html = """
    <html>
        <body>
            <div class="content">
                <p>First paragraph with enough text to meet the minimum length requirement for inclusion in results</p>
                <div>
                    <a href="https://example.com/page1">First link with enough text to be included in the output</a>
                    <p>Second paragraph that's also long enough to be included in the parsed output</p>
                    <a href="javascript:void(0)">JavaScript link to be ignored</a>
                    <a href="mailto:test@example.com">Email link to be ignored</a>
                    <a href="#section">Internal link to be ignored</a>
                </div>
            </div>
            <div class="ads">
                <script>
                    var adCode = 'ignored';
                </script>
                <iframe src="ad.html">Ignored ad content</iframe>
            </div>
        </body>
    </html>
    """
    result = parse_html(html)
    assert "First paragraph with enough text" in result
    assert "Second paragraph that's also long enough" in result
    assert "[First link with enough text to be included in the output](https://example.com/page1)" in result
    assert "JavaScript link" not in result
    assert "Email link" not in result
    assert "Internal link" not in result
    assert "adCode" not in result
    assert "Ignored ad content" not in result

@pytest.mark.asyncio
async def test_fetch_page():
    """Test page fetching"""
    # Test invalid URL
    with pytest.raises(ValidationError):
        await fetch_page("not-a-url")
    
    # Test successful fetch
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="Test content")
    
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=AsyncContextManagerMock(mock_response))
    
    content = await fetch_page("http://example.com", session=mock_session)
    assert content == "Test content"
    
    # Test HTTP error
    mock_response.status = 404
    mock_session.get = MagicMock(return_value=AsyncContextManagerMock(mock_response))
    
    with pytest.raises(FetchError) as exc_info:
        await fetch_page("http://example.com", session=mock_session)
    assert "HTTP error" in str(exc_info.value)
    assert "404" in str(exc_info.value)
    
    # Test timeout error
    mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
    with pytest.raises(FetchError) as exc_info:
        await fetch_page("http://example.com", session=mock_session)
    assert "Request timed out" in str(exc_info.value)
    
    # Test network error
    mock_session.get = MagicMock(side_effect=aiohttp.ClientError("Network error"))
    with pytest.raises(FetchError) as exc_info:
        await fetch_page("http://example.com", session=mock_session)
    assert "Network error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_process_urls():
    """Test URL processing"""
    # Test with invalid URLs
    with pytest.raises(ValidationError):
        await process_urls(["not-a-url"])
    
    # Test with empty URL list
    with pytest.raises(ValidationError):
        await process_urls([])
    
    # Test successful processing
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="""
        <html>
            <body>
                <p>Test content that is long enough to be included in the results</p>
            </body>
        </html>
    """)
    
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=AsyncContextManagerMock(mock_response))
    
    urls = ["http://example.com", "http://example.org"]
    result = await process_urls(urls, session=mock_session)
    
    assert len(result["results"]) == 2
    assert len(result["errors"]) == 0
    assert "Test content" in result["results"][0]["content"]
    
    # Test mixed success and failure
    def mock_get(url, **kwargs):
        if url == "http://example.com":
            response = MagicMock()
            response.status = 200
            response.text = AsyncMock(return_value="""
                <html>
                    <body>
                        <p>This is a longer piece of content that should definitely meet the minimum length requirement for inclusion in the results. We want to make sure it's processed correctly.</p>
                    </body>
                </html>
            """)
            return AsyncContextManagerMock(response)
        else:
            response = MagicMock()
            response.status = 404
            return AsyncContextManagerMock(response)
    
    mock_session.get = mock_get
    result = await process_urls(urls, session=mock_session)
    
    assert len(result["results"]) == 1
    assert len(result["errors"]) == 1
    assert "longer piece of content" in result["results"][0]["content"]
    assert "http://example.org" in result["errors"]
    
    # Test max_concurrent limit
    urls = [f"http://example.com/{i}" for i in range(5)]
    result = await process_urls(urls, max_concurrent=2, session=mock_session)
    assert len(result["results"]) + len(result["errors"]) == 5

def test_format_output():
    """Test output formatting"""
    results = [
        {"url": "http://example.com", "content": "Example text", "timestamp": 123456789},
        {"url": "http://failed.com", "content": "", "timestamp": 123456789}
    ]
    errors = {
        "http://error.com": "Failed to fetch"
    }
    data = {
        "results": results,
        "errors": errors
    }
    
    # Test text format
    text_output = format_output(data, "text", "Web Scraping Results")
    assert "Example text" in text_output
    assert "Failed to fetch" in text_output
    
    # Test JSON format
    json_output = format_output(data, "json", "Web Scraping Results")
    parsed = json.loads(json_output)
    assert len(parsed["data"]["results"]) == 2
    assert parsed["data"]["results"][0]["url"] == "http://example.com"
    assert parsed["title"] == "Web Scraping Results"
    
    # Test markdown format
    md_output = format_output(data, "markdown", "Web Scraping Results")
    assert "# Web Scraping Results" in md_output
    assert "http://example.com" in md_output
