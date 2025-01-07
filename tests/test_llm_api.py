import unittest
from unittest.mock import patch, MagicMock
from tools.llm_api import create_llm_client, query_llm
import os

def is_llm_configured():
    """Check if LLM is configured by trying to connect to the server"""
    try:
        client = create_llm_client()
        response = query_llm("test", client)
        return response is not None
    except:
        return False

# Skip all LLM tests if LLM is not configured
skip_llm_tests = not is_llm_configured()
skip_message = "Skipping LLM tests as LLM is not configured. This is normal if you haven't set up a local LLM server."

class TestLLMAPI(unittest.TestCase):
    def setUp(self):
        # Create a mock OpenAI client
        self.mock_client = MagicMock()
        self.mock_response = MagicMock()
        self.mock_choice = MagicMock()
        self.mock_message = MagicMock()
        
        # Set up the mock response structure
        self.mock_message.content = "Test response"
        self.mock_choice.message = self.mock_message
        self.mock_response.choices = [self.mock_choice]
        
        # Set up the mock client's chat.completions.create method
        self.mock_client.chat.completions.create.return_value = self.mock_response

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.OpenAI')
    def test_create_llm_client(self, mock_openai):
        # Test client creation with default provider (openai)
        mock_openai.return_value = self.mock_client
        client = create_llm_client()  # 使用預設 provider
        
        # Verify OpenAI was called with correct parameters
        mock_openai.assert_called_once_with(
            api_key=os.getenv('OPENAI_API_KEY')  # 使用環境變數中的 API key
        )
        
        self.assertEqual(client, self.mock_client)

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_llm_success(self, mock_create_client):
        # Set up mock
        mock_create_client.return_value = self.mock_client
        
        # Test query with default provider
        response = query_llm("Test prompt")  # 使用預設 provider
        
        # Verify response
        self.assertEqual(response, "Test response")
        
        # Verify client was called correctly
        self.mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",  # 使用 OpenAI 的預設模型
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_llm_with_custom_model(self, mock_create_client):
        # Set up mock
        mock_create_client.return_value = self.mock_client
        
        # Test query with custom model
        response = query_llm("Test prompt", model="custom-model")
        
        # Verify response
        self.assertEqual(response, "Test response")
        
        # Verify client was called with custom model
        self.mock_client.chat.completions.create.assert_called_once_with(
            model="custom-model",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_llm_with_existing_client(self, mock_create_client):
        # Test query with provided client
        response = query_llm("Test prompt", client=self.mock_client)
        
        # Verify response
        self.assertEqual(response, "Test response")
        
        # Verify create_client was not called
        mock_create_client.assert_not_called()

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_llm_error(self, mock_create_client):
        # Set up mock to raise an exception
        self.mock_client.chat.completions.create.side_effect = Exception("Test error")
        mock_create_client.return_value = self.mock_client
        
        # Test query with error
        response = query_llm("Test prompt")
        
        # Verify error handling
        self.assertIsNone(response)

if __name__ == '__main__':
    unittest.main()
