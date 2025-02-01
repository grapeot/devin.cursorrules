#!/usr/bin/env python3

import os
import pytest
from unittest.mock import patch, MagicMock, mock_open, AsyncMock
from tools.screenshot_utils import take_screenshot_sync, take_screenshot
from tools.llm_api import query_llm
from tools.token_tracker import TokenUsage

class TestScreenshotVerification:
    @pytest.fixture
    def mock_page(self):
        """Mock Playwright page object."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.screenshot = AsyncMock()
        mock_page.set_viewport_size = AsyncMock()
        return mock_page
    
    @pytest.fixture
    def mock_context(self, mock_page):
        """Mock Playwright browser context."""
        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        return mock_context
    
    @pytest.fixture
    def mock_browser(self, mock_page):
        """Mock Playwright browser."""
        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)
        mock_browser.close = AsyncMock()
        return mock_browser
    
    @pytest.fixture
    def mock_playwright(self, mock_browser):
        """Mock Playwright instance."""
        mock_playwright = AsyncMock()
        mock_playwright.chromium = AsyncMock()
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        return mock_playwright
    
    def test_screenshot_capture(self, mock_playwright, mock_page, tmp_path):
        """Test screenshot capture functionality with mocked Playwright."""
        # Ensure the output directory exists
        os.makedirs(tmp_path, exist_ok=True)
        output_path = os.path.join(tmp_path, 'test_screenshot.png')
        
        # Create a mock file to simulate screenshot being written
        with open(output_path, 'wb') as f:
            f.write(b'fake_screenshot_data')
        
        # Mock the async_playwright function and ensure the mock chain is connected
        with patch('tools.screenshot_utils.async_playwright', return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_playwright),
            __aexit__=AsyncMock()
        )):
            # Take the screenshot
            actual_path = take_screenshot_sync('http://test.com', output_path)
            
            # Verify the path is correct
            assert actual_path == output_path
            # Verify the file exists and has content
            assert os.path.exists(actual_path)
            with open(actual_path, 'rb') as f:
                assert f.read() == b'fake_screenshot_data'
            
            # Verify the mock chain was called correctly
            mock_playwright.chromium.launch.assert_called_once_with(headless=True)
            mock_browser = mock_playwright.chromium.launch.return_value
            mock_browser.new_page.assert_called_once_with(viewport={'width': 1280, 'height': 720})
            mock_page.goto.assert_called_once_with('http://test.com', wait_until='networkidle')
            mock_page.screenshot.assert_called_once_with(path=output_path, full_page=True)
            mock_browser.close.assert_called_once()
    
    def test_llm_verification_openai(self, tmp_path):
        """Test screenshot verification with OpenAI using mocks."""
        screenshot_path = os.path.join(tmp_path, 'test_screenshot.png')
        
        # Create a dummy screenshot file
        os.makedirs(tmp_path, exist_ok=True)
        with open(screenshot_path, 'wb') as f:
            f.write(b'fake_screenshot_data')
        
        # Mock the entire OpenAI client chain
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "The webpage has a blue background and the title is 'agentic.ai test page'"
        
        # Set up token usage with proper object-like attributes
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.completion_tokens_details = MagicMock()
        mock_usage.completion_tokens_details.reasoning_tokens = None
        mock_response.usage = mock_usage
        
        mock_openai.chat.completions.create.return_value = mock_response
        
        with patch('tools.llm_api.create_llm_client', return_value=mock_openai):
            response = query_llm(
                "What is the background color of this webpage? What is the title?",
                provider="openai",
                image_path=screenshot_path
            )
            
            assert 'blue' in response.lower()
            assert 'agentic.ai test page' in response.lower()
            mock_openai.chat.completions.create.assert_called_once()
    
    def test_llm_verification_anthropic(self, tmp_path):
        """Test screenshot verification with Anthropic using mocks."""
        screenshot_path = os.path.join(tmp_path, 'test_screenshot.png')
        
        # Create a dummy screenshot file
        os.makedirs(tmp_path, exist_ok=True)
        with open(screenshot_path, 'wb') as f:
            f.write(b'fake_screenshot_data')
        
        # Mock the entire Anthropic client chain
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "The webpage has a blue background and the title is 'agentic.ai test page'"
        mock_response.content = [mock_content]
        
        # Set up token usage with proper object-like attributes
        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_response.usage = mock_usage
        
        mock_anthropic.messages.create.return_value = mock_response
        
        with patch('tools.llm_api.create_llm_client', return_value=mock_anthropic):
            response = query_llm(
                "What is the background color of this webpage? What is the title?",
                provider="anthropic",
                image_path=screenshot_path
            )
            
            assert 'blue' in response.lower()
            assert 'agentic.ai test page' in response.lower()
            mock_anthropic.messages.create.assert_called_once()

# Note: End-to-end tests have been moved to tools/test_e2e.py 