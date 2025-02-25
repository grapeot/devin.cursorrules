#!/usr/bin/env python3

import unittest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add the parent directory to the Python path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.plan_exec_llm import load_environment, read_plan_status, read_file_content, query_llm_with_plan
from tools.token_tracker import TokenUsage

class TestPlanExecLLM(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Save original environment
        self.original_env = dict(os.environ)
        # Set test environment variables
        os.environ['OPENAI_API_KEY'] = 'test_key'
        os.environ['DEEPSEEK_API_KEY'] = 'test_deepseek_key'
        os.environ['ANTHROPIC_API_KEY'] = 'test_anthropic_key'
        
        self.test_env_content = """
OPENAI_API_KEY=test_key
DEEPSEEK_API_KEY=test_deepseek_key
ANTHROPIC_API_KEY=test_anthropic_key
"""
        self.test_plan_content = """
# Multi-Agent Scratchpad
Test content
"""
        # Create temporary test files
        with open('.env.test', 'w') as f:
            f.write(self.test_env_content)
        with open('.cursorrules.test', 'w') as f:
            f.write(self.test_plan_content)

    def tearDown(self):
        """Clean up test fixtures"""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Remove temporary test files
        for file in ['.env.test', '.cursorrules.test']:
            if os.path.exists(file):
                os.remove(file)

    @patch('tools.plan_exec_llm.load_dotenv')
    def test_load_environment(self, mock_load_dotenv):
        """Test environment loading"""
        load_environment()
        mock_load_dotenv.assert_called()

    def test_read_plan_status(self):
        """Test reading plan status"""
        with patch('tools.plan_exec_llm.STATUS_FILE', '.cursorrules.test'):
            content = read_plan_status()
            self.assertIn('# Multi-Agent Scratchpad', content)
            self.assertIn('Test content', content)

    def test_read_file_content(self):
        """Test reading file content"""
        # Test with existing file
        content = read_file_content('.env.test')
        self.assertIn('OPENAI_API_KEY=test_key', content)

        # Test with non-existent file
        content = read_file_content('nonexistent_file.txt')
        self.assertIsNone(content)

    @patch('tools.llm_api.query_llm')
    def test_query_llm_with_plan(self, mock_query_llm):
        """Test LLM querying with plan context"""
        # Mock the LLM response
        mock_query_llm.return_value = "Test response"

        # Test with various combinations of parameters
        with patch('tools.plan_exec_llm.query_llm') as mock_plan_query_llm:
            mock_plan_query_llm.return_value = "Test response"
            response = query_llm_with_plan("Test plan", "Test prompt", "Test file content", provider="openai", model="gpt-4o")
            self.assertEqual(response, "Test response")
            mock_plan_query_llm.assert_called_with(unittest.mock.ANY, model="gpt-4o", provider="openai")

            # Test with DeepSeek
            response = query_llm_with_plan("Test plan", "Test prompt", provider="deepseek")
            self.assertEqual(response, "Test response")
            mock_plan_query_llm.assert_called_with(unittest.mock.ANY, model=None, provider="deepseek")

            # Test with Anthropic
            response = query_llm_with_plan("Test plan", provider="anthropic")
            self.assertEqual(response, "Test response")
            mock_plan_query_llm.assert_called_with(unittest.mock.ANY, model=None, provider="anthropic")

            # Verify the prompt format
            calls = mock_plan_query_llm.call_args_list
            for call in calls:
                prompt = call[0][0]
                self.assertIn("Multi-Agent Scratchpad", prompt)
                self.assertIn("Test plan", prompt)

if __name__ == '__main__':
    unittest.main() 