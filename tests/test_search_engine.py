import unittest
from unittest.mock import patch, MagicMock
import sys
from io import StringIO
from tools.search_engine import (
    search,
    SearchError,
    get_search_engine,
    DuckDuckGoEngine,
    GoogleEngine,
    fetch_page_snippet
)
from tools.common.errors import ValidationError
import pytest

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()
        patch('tools.search_engine.logger', self.mock_logger).start()

    def tearDown(self):
        patch.stopall()

    @patch('tools.search_engine.requests.get')
    def test_fetch_page_snippet(self, mock_get):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.text = '''
            <html>
                <head>
                    <title>Test Title</title>
                    <meta name="description" content="Meta description snippet">
                </head>
                <body>
                    <p>First paragraph</p>
                    <p>Second paragraph with enough text to be considered a valid snippet candidate that should be selected for the result</p>
                </body>
            </html>
        '''
        mock_get.return_value = mock_response
        
        # Test with meta description
        title, snippet = fetch_page_snippet("http://example.com", "test query")
        self.assertEqual(title, "Test Title")
        self.assertEqual(snippet, "Meta description snippet")
        
        # Test without meta description
        mock_response.text = '''
            <html>
                <head>
                    <title>Test Title</title>
                </head>
                <body>
                    <p>Short text</p>
                    <p>Second paragraph with enough text to be considered a valid snippet candidate that should be selected for the result</p>
                </body>
            </html>
        '''
        title, snippet = fetch_page_snippet("http://example.com", "test query")
        self.assertEqual(title, "Test Title")
        self.assertTrue(snippet.startswith("Second paragraph"))
        
        # Test error handling
        mock_get.side_effect = Exception("Connection error")
        with self.assertRaises(SearchError) as cm:
            fetch_page_snippet("http://example.com", "test query")
        self.assertIn("Connection error", str(cm.exception))

    def test_get_search_engine(self):
        # Test valid engines
        engine = get_search_engine("duckduckgo")
        self.assertIsInstance(engine, DuckDuckGoEngine)
        
        engine = get_search_engine("google")
        self.assertIsInstance(engine, GoogleEngine)
        
        # Test case insensitivity
        engine = get_search_engine("DUCKDUCKGO")
        self.assertIsInstance(engine, DuckDuckGoEngine)
        
        # Test invalid engine
        with pytest.raises(ValidationError) as exc_info:
            get_search_engine("invalid")
        assert "Invalid search engine" in str(exc_info.value)

    @patch('tools.search_engine.DDGS')
    def test_duckduckgo_search(self, mock_ddgs):
        # Mock search results
        mock_results = [
            {
                'href': 'http://example.com',
                'title': 'Example Title',
                'body': 'Example Body'
            },
            {
                'href': 'http://example2.com',
                'title': 'Example Title 2',
                'body': 'Example Body 2'
            }
        ]
        
        # Setup mock
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Run search
        results = search("test query", max_results=2, engine="duckduckgo")
        
        # Check logging
        self.mock_logger.info.assert_any_call(
            "Searching for: test query",
            extra={
                "context": {
                    "engine": "duckduckgo",
                    "max_results": 2,
                    "max_retries": 3,
                    "fetch_snippets": True
                }
            }
        )
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["url"], "http://example.com")
        self.assertEqual(results[0]["title"], "Example Title")
        self.assertEqual(results[0]["snippet"], "Example Body")
        
    @patch('tools.search_engine.google_search')
    @patch('tools.search_engine.fetch_page_snippet')
    def test_google_search_with_snippets(self, mock_fetch_snippet, mock_google):
        # Mock search results
        mock_google.return_value = [
            'http://example.com',
            'http://example2.com'
        ]
        
        # Mock snippet fetching
        mock_fetch_snippet.side_effect = [
            ("Example Title", "Example Snippet"),
            ("Example Title 2", "Example Snippet 2")
        ]
        
        # Run search with snippet fetching
        results = search("test query", max_results=2, engine="google", fetch_snippets=True)
        
        # Check logging
        self.mock_logger.info.assert_any_call(
            "Searching for: test query",
            extra={
                "context": {
                    "engine": "google",
                    "max_results": 2,
                    "max_retries": 3,
                    "fetch_snippets": True
                }
            }
        )
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["url"], "http://example.com")
        self.assertEqual(results[0]["title"], "Example Title")
        self.assertEqual(results[0]["snippet"], "Example Snippet")
        
    @patch('tools.search_engine.google_search')
    @patch('tools.search_engine.fetch_page_snippet')
    def test_google_search_without_snippets(self, mock_fetch_snippet, mock_google):
        # Mock search results
        mock_google.return_value = [
            'http://example.com',
            'http://example2.com'
        ]

        # Run search without snippet fetching
        results = search("test query", max_results=2, engine="google", fetch_snippets=False)

        # Check search results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["url"], "http://example.com")
        self.assertEqual(results[0]["title"], "http://example.com")
        self.assertEqual(results[0]["snippet"], "")
        self.assertEqual(results[1]["url"], "http://example2.com")
        self.assertEqual(results[1]["title"], "http://example2.com")
        self.assertEqual(results[1]["snippet"], "")

        # Verify fetch_snippet was not called
        mock_fetch_snippet.assert_not_called()

    @patch('tools.search_engine.google_search')
    @patch('tools.search_engine.fetch_page_snippet')
    def test_google_search_with_failed_snippets(self, mock_fetch_snippet, mock_google):
        # Mock search results
        mock_google.return_value = [
            'http://example.com',
            'http://example2.com'
        ]
        
        # Mock snippet fetching with failures
        mock_fetch_snippet.side_effect = [
            ("Example Title", "Example Snippet"),
            SearchError("Failed to fetch", "google", "test query")
        ]

        # Run search
        results = search("test query", max_results=2, engine="google")

        # Check results - should include both successful and failed snippets
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["url"], "http://example.com")
        self.assertEqual(results[0]["title"], "Example Title")
        self.assertEqual(results[0]["snippet"], "Example Snippet")
        self.assertEqual(results[1]["url"], "http://example2.com")
        self.assertEqual(results[1]["title"], "http://example2.com")
        self.assertEqual(results[1]["snippet"], "")

        # Verify warning was logged
        self.mock_logger.warning.assert_called_with(
            "Failed to fetch snippet for http://example2.com: Failed to fetch (query=test query, provider=google)"
        )

    @patch('tools.search_engine.DDGS')
    def test_no_results(self, mock_ddgs):
        # Mock empty results
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.return_value = []
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Run search
        results = search("test query")
        
        # Check logging
        self.mock_logger.info.assert_any_call(
            "Searching for: test query",
            extra={
                "context": {
                    "engine": "duckduckgo",
                    "max_results": 10,
                    "max_retries": 3,
                    "fetch_snippets": True
                }
            }
        )
        
        # Check results
        self.assertEqual(len(results), 0)

    @patch('tools.search_engine.DDGS')
    def test_search_error(self, mock_ddgs):
        # Mock search error
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.side_effect = Exception("Test error")
        mock_ddgs.return_value = mock_ddgs_instance

        # Run search and check for error
        with pytest.raises(SearchError) as exc_info:
            search("test query")
        
        assert "DuckDuckGo search failed" in str(exc_info.value)
        assert "Test error" in str(exc_info.value)

    def test_invalid_inputs(self):
        # Test empty query
        with pytest.raises(ValidationError) as exc_info:
            search("")
        assert "Search query cannot be empty" in str(exc_info.value)

        # Test whitespace query
        with pytest.raises(ValidationError) as exc_info:
            search("   ")
        assert "Search query cannot be empty" in str(exc_info.value)

        # Test invalid max_results
        with pytest.raises(ValidationError) as exc_info:
            search("test", max_results=0)
        assert "max_results must be a positive integer" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            search("test", max_results=101)
        assert "max_results cannot exceed 100" in str(exc_info.value)

        # Test invalid engine
        with pytest.raises(ValidationError) as exc_info:
            search("test", engine="invalid")
        assert "Invalid search engine" in str(exc_info.value)

if __name__ == '__main__':
    unittest.main()
