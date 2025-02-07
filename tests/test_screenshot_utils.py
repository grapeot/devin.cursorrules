#!/usr/bin/env python3

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import tempfile
from pathlib import Path
import pytest
import asyncio
from playwright.async_api import Error as PlaywrightError
from tools.screenshot_utils import (
    validate_url,
    validate_dimensions,
    take_screenshot,
    take_screenshot_sync,
    ScreenshotError
)
from tools.common.errors import ValidationError, FileError

class AsyncContextManagerMock:
    def __init__(self, response):
        self.response = response
    
    async def __aenter__(self):
        return self.response
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class TestScreenshotUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_url = "https://example.com"
        self.test_output = "test_screenshot.png"
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "screenshot.png")

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary files
        if os.path.exists(self.temp_file):
            os.unlink(self.temp_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_validate_url(self):
        """Test URL validation"""
        # Test valid URLs
        self.assertTrue(validate_url("http://example.com"))
        self.assertTrue(validate_url("https://example.com"))
        self.assertTrue(validate_url("https://sub.example.com/path?query=1"))
        
        # Test invalid URLs
        self.assertFalse(validate_url(""))
        self.assertFalse(validate_url("not-a-url"))
        self.assertFalse(validate_url("ftp://example.com"))
        self.assertFalse(validate_url("http://"))
        self.assertFalse(validate_url("https://"))

    def test_validate_dimensions(self):
        """Test viewport dimension validation"""
        # Test valid dimensions
        self.assertEqual(validate_dimensions(800, 600), (800, 600))
        self.assertEqual(validate_dimensions(1, 1), (1, 1))
        self.assertEqual(validate_dimensions(16383, 16383), (16383, 16383))
        
        # Test invalid dimensions
        with self.assertRaises(ValidationError) as cm:
            validate_dimensions(0, 600)
        self.assertIn("must be positive", str(cm.exception))
        
        with self.assertRaises(ValidationError) as cm:
            validate_dimensions(800, 0)
        self.assertIn("must be positive", str(cm.exception))
        
        with self.assertRaises(ValidationError) as cm:
            validate_dimensions(-1, 600)
        self.assertIn("must be positive", str(cm.exception))
        
        with self.assertRaises(ValidationError) as cm:
            validate_dimensions(800, -1)
        self.assertIn("must be positive", str(cm.exception))
        
        with self.assertRaises(ValidationError) as cm:
            validate_dimensions(16385, 600)
        self.assertIn("cannot exceed 16384", str(cm.exception))
        
        with self.assertRaises(ValidationError) as cm:
            validate_dimensions(800, 16385)
        self.assertIn("cannot exceed 16384", str(cm.exception))

@pytest.mark.asyncio
async def test_take_screenshot_validation():
    """Test screenshot taking input validation"""
    # Test invalid URL
    with pytest.raises(ValidationError):
        await take_screenshot("not-a-url")
    
    # Test invalid dimensions
    with pytest.raises(ValidationError):
        await take_screenshot("https://example.com", width=0)
    with pytest.raises(ValidationError):
        await take_screenshot("https://example.com", height=0)

@pytest.mark.asyncio
@patch('tools.screenshot_utils.async_playwright')
async def test_take_screenshot_success(mock_playwright):
    """Test successful screenshot capture"""
    # Mock Playwright objects
    mock_browser = AsyncMock()
    mock_page = AsyncMock()
    mock_context = AsyncMock()
    
    # Set up mock chain
    mock_playwright.return_value = AsyncContextManagerMock(mock_context)
    mock_context.chromium = AsyncMock()
    mock_context.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_page.goto = AsyncMock()
    
    # Mock screenshot to create a file
    async def mock_screenshot(path, **kwargs):
        with open(path, 'wb') as f:
            f.write(b'fake screenshot data')
    mock_page.screenshot = AsyncMock(side_effect=mock_screenshot)
    
    # Test with default output path (temporary file)
    result = await take_screenshot("https://example.com")
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0
    os.unlink(result)  # Clean up temp file
    
    # Test with specified output path
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "screenshot.png")
    result = await take_screenshot("https://example.com", temp_file)
    assert result == temp_file
    assert os.path.exists(temp_file)
    assert os.path.getsize(temp_file) > 0
    
    # Verify calls
    mock_context.chromium.launch.assert_called_with(headless=True)
    mock_browser.new_page.assert_called_with(viewport={'width': 1280, 'height': 720})
    mock_page.goto.assert_called_with("https://example.com", wait_until='networkidle')
    mock_page.screenshot.assert_called_with(path=temp_file, full_page=True)
    
    # Test with custom dimensions
    await take_screenshot("https://example.com", temp_file, width=800, height=600)
    mock_browser.new_page.assert_called_with(viewport={'width': 800, 'height': 600})
    
    # Clean up
    os.unlink(temp_file)
    os.rmdir(temp_dir)

@pytest.mark.asyncio
@patch('tools.screenshot_utils.async_playwright')
async def test_take_screenshot_failure(mock_playwright):
    """Test screenshot capture failures"""
    # Mock Playwright error
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_playwright.return_value = AsyncContextManagerMock(mock_context)
    mock_context.chromium = AsyncMock()
    mock_context.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_browser.new_page = AsyncMock(side_effect=PlaywrightError("Test error"))
    
    # Test screenshot failure
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "screenshot.png")
    with pytest.raises(ScreenshotError) as cm:
        await take_screenshot("https://example.com", temp_file)
    assert "Test error" in str(cm.value)
    assert cm.value.url == "https://example.com"
    assert not os.path.exists(temp_file)  # File should not be created on error
    
    # Test output directory creation failure
    bad_path = "/nonexistent/dir/screenshot.png"
    with pytest.raises(FileError) as cm:
        await take_screenshot("https://example.com", bad_path)
    assert "Failed to create output directory" in str(cm.value)
    
    # Test temporary file creation failure
    with patch('tempfile.NamedTemporaryFile', side_effect=OSError("Test error")):
        with pytest.raises(FileError) as cm:
            await take_screenshot("https://example.com")
        assert "Failed to create temporary file" in str(cm.value)
    
    # Clean up
    os.rmdir(temp_dir)

def test_take_screenshot_sync():
    """Test synchronous screenshot capture"""
    # Test invalid URL
    with pytest.raises(ValidationError):
        take_screenshot_sync("not-a-url")
    
    # Test invalid dimensions
    with pytest.raises(ValidationError):
        take_screenshot_sync("https://example.com", width=0)
    with pytest.raises(ValidationError):
        take_screenshot_sync("https://example.com", height=0)

@patch('tools.screenshot_utils.asyncio.run')
def test_take_screenshot_sync_error_handling(mock_run):
    """Test error handling in synchronous screenshot capture"""
    # Test ScreenshotError passthrough
    mock_run.side_effect = ScreenshotError("Test error", "https://example.com")
    with pytest.raises(ScreenshotError) as cm:
        take_screenshot_sync("https://example.com")
    assert "Test error" in str(cm.value)
    
    # Test ValidationError passthrough
    mock_run.side_effect = ValidationError("Test error")
    with pytest.raises(ValidationError):
        take_screenshot_sync("https://example.com")
    
    # Test FileError passthrough
    mock_run.side_effect = FileError("Test error", "test.png")
    with pytest.raises(FileError):
        take_screenshot_sync("https://example.com")
    
    # Test other exceptions
    mock_run.side_effect = Exception("Unexpected error")
    with pytest.raises(ScreenshotError) as cm:
        take_screenshot_sync("https://example.com")
    assert "Unexpected error" in str(cm.value)

if __name__ == '__main__':
    unittest.main() 