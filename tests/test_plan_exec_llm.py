#!/usr/bin/env python3

import unittest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add the parent directory to the Python path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.plan_exec_llm import load_environment, read_plan_status, read_file_content, create_llm_client, query_llm
from tools.plan_exec_llm import TokenUsage

class TestPlanExecLLM(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_env_content = """
OPENAI_API_KEY=test_key
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

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_create_llm_client(self):
        """Test LLM client creation"""
        client = create_llm_client()
        self.assertIsNotNone(client)

    @patch('tools.plan_exec_llm.OpenAI')
    def test_query_llm(self, mock_openai):
        """Test LLM querying"""
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            reasoning_tokens=None
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Test with various combinations of parameters
        response = query_llm("Test plan", "Test prompt", "Test file content")
        self.assertEqual(response, "Test response")

        response = query_llm("Test plan", "Test prompt")
        self.assertEqual(response, "Test response")

        response = query_llm("Test plan")
        self.assertEqual(response, "Test response")

        # Verify the OpenAI client was called with correct parameters
        mock_client.chat.completions.create.assert_called_with(
            model="o1",
            messages=[
                {"role": "system", "content": unittest.mock.ANY},
                {"role": "user", "content": unittest.mock.ANY}
            ],
            response_format={"type": "text"},
            reasoning_effort="low"
        )

if __name__ == '__main__':
    unittest.main() 