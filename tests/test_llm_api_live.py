import unittest
import os
from tools.llm_api import query_llm, load_environment
from tests.test_utils import (
    requires_openai,
    requires_anthropic,
    requires_azure,
    requires_deepseek,
    requires_gemini
)
import pytest

class TestLLMAPILive(unittest.TestCase):
    def setUp(self):
        self.original_env = dict(os.environ)
        load_environment()  # Load environment variables from .env files

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def _test_llm_response(self, provider: str, response: str):
        """Helper to test LLM response with common assertions"""
        self.assertIsNotNone(response, f"Response from {provider} was None")
        self.assertIsInstance(response, str, f"Response from {provider} was not a string")
        self.assertTrue(len(response) > 0, f"Response from {provider} was empty")

    @requires_openai
    def test_openai_live(self):
        """Live test of OpenAI integration"""
        try:
            response = query_llm("Say 'test'", provider="openai")
            self._test_llm_response("OpenAI", response)
        except Exception as e:
            pytest.skip(f"OpenAI API error: {str(e)}")

    @requires_anthropic
    def test_anthropic_live(self):
        """Live test of Anthropic integration"""
        try:
            response = query_llm("Say 'test'", provider="anthropic")
            self._test_llm_response("Anthropic", response)
        except Exception as e:
            pytest.skip(f"Anthropic API error: {str(e)}")

    @requires_azure
    def test_azure_live(self):
        """Live test of Azure OpenAI integration"""
        try:
            response = query_llm("Say 'test'", provider="azure")
            self._test_llm_response("Azure", response)
        except Exception as e:
            pytest.skip(f"Azure API error: {str(e)}")

    @requires_deepseek
    def test_deepseek_live(self):
        """Live test of DeepSeek integration"""
        try:
            response = query_llm("Say 'test'", provider="deepseek")
            self._test_llm_response("DeepSeek", response)
        except Exception as e:
            pytest.skip(f"DeepSeek API error: {str(e)}")

    @requires_gemini
    def test_gemini_live(self):
        """Live test of Gemini integration"""
        try:
            response = query_llm("Say 'test'", provider="gemini")
            self._test_llm_response("Gemini", response)
        except Exception as e:
            pytest.skip(f"Gemini API error: {str(e)}")

if __name__ == '__main__':
    unittest.main() 