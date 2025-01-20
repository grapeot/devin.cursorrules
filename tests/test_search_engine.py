import unittest
from unittest.mock import patch, MagicMock
import sys
from io import StringIO
from tools.search_engine import search

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        # Capture stdout and stderr for testing
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def tearDown(self):
        # Restore stdout and stderr
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

    @patch('tools.search_engine.DDGS')
    def test_successful_search(self, mock_ddgs):
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
        search("test query", max_results=2)

        # Check debug output
        expected_debug = "DEBUG: Searching for query: test query (attempt 1/3)"
        self.assertIn(expected_debug, self.stderr.getvalue())
        self.assertIn("DEBUG: Found 2 results", self.stderr.getvalue())

        # Check search results output
        output = self.stdout.getvalue()
        self.assertIn("=== Result 1 ===", output)
        self.assertIn("URL: http://example.com", output)
        self.assertIn("Title: Example Title", output)
        self.assertIn("Snippet: Example Body", output)
        self.assertIn("=== Result 2 ===", output)
        self.assertIn("URL: http://example2.com", output)
        self.assertIn("Title: Example Title 2", output)
        self.assertIn("Snippet: Example Body 2", output)

        # Verify mock was called correctly
        mock_ddgs_instance.__enter__.return_value.text.assert_called_once_with(
            "test query",
            max_results=2
        )

    @patch('tools.search_engine.DDGS')
    def test_no_results(self, mock_ddgs):
        # Mock empty results
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.return_value = []
        mock_ddgs.return_value = mock_ddgs_instance

        # Run search
        search("test query")

        # Check debug output
        self.assertIn("DEBUG: No results found", self.stderr.getvalue())

        # Check that no results were printed
        self.assertEqual("", self.stdout.getvalue().strip())

    @patch('tools.search_engine.DDGS')
    def test_search_error(self, mock_ddgs):
        # Mock search error
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.side_effect = Exception("Test error")
        mock_ddgs.return_value = mock_ddgs_instance

        # Run search and check for error
        with self.assertRaises(SystemExit) as cm:
            search("test query")
        
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("ERROR: Search failed: Test error", self.stderr.getvalue())

    def test_result_field_fallbacks(self):
        # Test that the fields work correctly with N/A fallback
        result = {
            'href': 'http://example.com',
            'title': 'Example Title',
            'body': 'Example Body'
        }
        
        # Test fields present
        self.assertEqual(result.get('href', 'N/A'), 'http://example.com')
        self.assertEqual(result.get('title', 'N/A'), 'Example Title')
        self.assertEqual(result.get('body', 'N/A'), 'Example Body')
        
        # Test missing fields
        result = {}
        self.assertEqual(result.get('href', 'N/A'), 'N/A')
        self.assertEqual(result.get('title', 'N/A'), 'N/A')
        self.assertEqual(result.get('body', 'N/A'), 'N/A')

if __name__ == '__main__':
    unittest.main()
