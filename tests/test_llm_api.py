import unittest
from unittest.mock import patch, MagicMock, mock_open
from tools.llm_api import create_llm_client, query_llm, load_environment
from tools.token_tracker import TokenUsage, APIResponse, get_token_tracker
import os
import google.generativeai as genai
import io
import sys

class TestEnvironmentLoading(unittest.TestCase):
    def setUp(self):
        # Save original environment
        self.original_env = dict(os.environ)
        # Clear environment variables we're testing
        for key in ['TEST_VAR']:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch('pathlib.Path.exists')
    @patch('tools.llm_api.load_dotenv')
    @patch('builtins.open')
    def test_environment_loading_precedence(self, mock_open, mock_load_dotenv, mock_exists):
        # Mock all env files exist
        mock_exists.return_value = True
        
        # Mock file contents
        mock_file = MagicMock()
        mock_file.__enter__.return_value = io.StringIO('TEST_VAR=value\n')
        mock_open.return_value = mock_file
        
        # Mock different values for TEST_VAR in different files
        def load_dotenv_side_effect(dotenv_path, **kwargs):
            if '.env.local' in str(dotenv_path):
                os.environ['TEST_VAR'] = 'local'
            elif '.env' in str(dotenv_path):
                if 'TEST_VAR' not in os.environ:  # Only set if not already set
                    os.environ['TEST_VAR'] = 'default'
            elif '.env.example' in str(dotenv_path):
                if 'TEST_VAR' not in os.environ:  # Only set if not already set
                    os.environ['TEST_VAR'] = 'example'
        mock_load_dotenv.side_effect = load_dotenv_side_effect
        
        # Load environment
        load_environment()
        
        # Verify precedence (.env.local should win)
        self.assertEqual(os.environ.get('TEST_VAR'), 'local')
        
        # Verify order of loading
        calls = mock_load_dotenv.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertTrue(str(calls[0][1]['dotenv_path']).endswith('.env.local'))
        self.assertTrue(str(calls[1][1]['dotenv_path']).endswith('.env'))
        self.assertTrue(str(calls[2][1]['dotenv_path']).endswith('.env.example'))

    @patch('pathlib.Path.exists')
    @patch('tools.llm_api.load_dotenv')
    def test_environment_loading_no_files(self, mock_load_dotenv, mock_exists):
        # Mock no env files exist
        mock_exists.return_value = False
        
        # Load environment
        load_environment()
        
        # Verify load_dotenv was not called
        mock_load_dotenv.assert_not_called()

class TestLLMAPI(unittest.TestCase):
    def setUp(self):
        # Create mock clients for different providers
        self.mock_openai_client = MagicMock()
        self.mock_anthropic_client = MagicMock()
        self.mock_azure_client = MagicMock()
        self.mock_gemini_client = MagicMock()
        
        # Set up mock responses
        self.mock_openai_response = MagicMock()
        self.mock_openai_response.choices = [MagicMock()]
        self.mock_openai_response.choices[0].message = MagicMock()
        self.mock_openai_response.choices[0].message.content = "Test OpenAI response"
        self.mock_openai_response.usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            reasoning_tokens=None
        )

        self.mock_anthropic_response = MagicMock()
        self.mock_anthropic_response.content = [MagicMock()]
        self.mock_anthropic_response.content[0].text = "Test Anthropic response"
        self.mock_anthropic_response.usage = MagicMock()
        self.mock_anthropic_response.usage.input_tokens = 10
        self.mock_anthropic_response.usage.output_tokens = 5

        self.mock_azure_response = MagicMock()
        self.mock_azure_response.choices = [MagicMock()]
        self.mock_azure_response.choices[0].message = MagicMock()
        self.mock_azure_response.choices[0].message.content = "Test Azure OpenAI response"
        self.mock_azure_response.usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            reasoning_tokens=None
        )
        
        # Set up return values for mock clients
        self.mock_openai_client.chat.completions.create.return_value = self.mock_openai_response
        self.mock_anthropic_client.messages.create.return_value = self.mock_anthropic_response
        self.mock_azure_client.chat.completions.create.return_value = self.mock_azure_response
        
        # Set up Gemini-style response
        self.mock_gemini_model = MagicMock()
        self.mock_gemini_response = MagicMock()
        self.mock_gemini_response.text = "Test Gemini response"
        self.mock_gemini_model.generate_content.return_value = self.mock_gemini_response
        self.mock_gemini_client.GenerativeModel.return_value = self.mock_gemini_model
        
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-openai-key',
            'DEEPSEEK_API_KEY': 'test-deepseek-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'GOOGLE_API_KEY': 'test-google-key',
            'AZURE_OPENAI_API_KEY': 'test-azure-key',
            'AZURE_OPENAI_MODEL_DEPLOYMENT': 'test-model-deployment'
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch('tools.llm_api.OpenAI')
    def test_create_openai_client(self, mock_openai):
        mock_openai.return_value = self.mock_openai_client
        client = create_llm_client("openai")
        mock_openai.assert_called_once_with(api_key='test-openai-key')
        self.assertEqual(client, self.mock_openai_client)

    @patch('tools.llm_api.AzureOpenAI')
    def test_create_azure_client(self, mock_azure):
        mock_azure.return_value = self.mock_azure_client
        client = create_llm_client("azure")
        mock_azure.assert_called_once_with(
            api_key='test-azure-key',
            api_version="2024-08-01-preview",
            azure_endpoint="https://msopenai.openai.azure.com"
        )
        self.assertEqual(client, self.mock_azure_client)

    @patch('tools.llm_api.OpenAI')
    def test_create_deepseek_client(self, mock_openai):
        mock_openai.return_value = self.mock_openai_client
        client = create_llm_client("deepseek")
        mock_openai.assert_called_once_with(
            api_key='test-deepseek-key',
            base_url="https://api.deepseek.com/v1"
        )
        self.assertEqual(client, self.mock_openai_client)

    @patch('tools.llm_api.Anthropic')
    def test_create_anthropic_client(self, mock_anthropic):
        mock_anthropic.return_value = self.mock_anthropic_client
        client = create_llm_client("anthropic")
        mock_anthropic.assert_called_once_with(api_key='test-anthropic-key')
        self.assertEqual(client, self.mock_anthropic_client)

    @patch('tools.llm_api.genai')
    def test_create_gemini_client(self, mock_genai):
        client = create_llm_client("gemini")
        mock_genai.configure.assert_called_once_with(api_key='test-google-key')
        self.assertEqual(client, mock_genai)

    def test_create_invalid_provider(self):
        with self.assertRaises(ValueError):
            create_llm_client("invalid_provider")

    @patch('tools.llm_api.OpenAI')
    def test_query_openai(self, mock_create_client):
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", provider="openai", model="gpt-4o")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @patch('tools.llm_api.create_llm_client')
    def test_query_azure(self, mock_create_client):
        mock_create_client.return_value = self.mock_azure_client
        response = query_llm("Test prompt", provider="azure", model="gpt-4o")
        self.assertEqual(response, "Test Azure OpenAI response")
        self.mock_azure_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @patch('tools.llm_api.create_llm_client')
    @patch('tools.llm_api.get_token_tracker')
    def test_query_deepseek(self, mock_get_tracker, mock_create_client):
        """Test querying DeepSeek API with token tracking.
        
        DeepSeek uses OpenAI-compatible API but like most models does not support
        reasoning tokens (only OpenAI's o1 model has this feature).
        """
        mock_create_client.return_value = self.mock_openai_client
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker
        
        # Set up mock response with usage data
        self.mock_openai_response.usage = MagicMock()
        self.mock_openai_response.usage.prompt_tokens = 10
        self.mock_openai_response.usage.completion_tokens = 5
        self.mock_openai_response.usage.total_tokens = 15
        
        response = query_llm("Test prompt", provider="deepseek", model="deepseek-chat")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )
        # Verify token usage tracking for OpenAI-style providers
        self.assertTrue(mock_tracker.track_request.called)
        api_response = mock_tracker.track_request.call_args[0][0]
        # Verify reasoning_tokens is None since this is not the o1 model
        self.assertIsNone(api_response.token_usage.reasoning_tokens)

    @patch('tools.llm_api.create_llm_client')
    def test_query_anthropic(self, mock_create_client):
        """Test querying Anthropic API.
        
        Note: Anthropic's API has its own token tracking system that differs from OpenAI's.
        It does not support reasoning tokens (which is an OpenAI o1-specific feature).
        """
        mock_create_client.return_value = self.mock_anthropic_client
        response = query_llm("Test prompt", provider="anthropic", model="claude-3-5-sonnet-20241022")
        self.assertEqual(response, "Test Anthropic response")
        self.mock_anthropic_client.messages.create.assert_called_once_with(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}]
        )
        # Note: Token tracking is not yet implemented for Anthropic

    @patch('tools.llm_api.create_llm_client')
    def test_query_gemini(self, mock_create_client):
        mock_create_client.return_value = self.mock_gemini_client
        response = query_llm("Test prompt", provider="gemini")
        self.assertEqual(response, "Test Gemini response")
        self.mock_gemini_client.GenerativeModel.assert_called_once_with("gemini-pro")
        self.mock_gemini_model.generate_content.assert_called_once_with("Test prompt")

    @patch('tools.llm_api.create_llm_client')
    def test_query_with_custom_model(self, mock_create_client):
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", provider="openai", model="gpt-4o")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @patch('tools.llm_api.create_llm_client')
    @patch('tools.llm_api.get_token_tracker')
    def test_query_o1_model(self, mock_get_tracker, mock_create_client):
        """Test querying OpenAI's o1 model.
        
        The o1 model is special in that it:
        1. Uses a different response format
        2. Has a reasoning_effort parameter
        3. Is the only model that provides reasoning_tokens in its response
        """
        mock_create_client.return_value = self.mock_openai_client
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker
        
        # Set up mock response with usage data including reasoning tokens
        self.mock_openai_response.usage = MagicMock()
        self.mock_openai_response.usage.prompt_tokens = 10
        self.mock_openai_response.usage.completion_tokens = 5
        self.mock_openai_response.usage.total_tokens = 15
        self.mock_openai_response.usage.reasoning_tokens = 3  # o1 model provides this
        
        response = query_llm("Test prompt", provider="openai", model="o1")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="o1",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            response_format={"type": "text"},
            reasoning_effort="low"
        )
        
        # Verify token usage tracking includes reasoning tokens for o1 model
        self.assertTrue(mock_tracker.track_request.called)
        api_response = mock_tracker.track_request.call_args[0][0]
        self.assertEqual(api_response.token_usage.reasoning_tokens, 3)

    @patch('tools.llm_api.create_llm_client')
    def test_query_with_existing_client(self, mock_create_client):
        response = query_llm("Test prompt", client=self.mock_openai_client, model="gpt-4o")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @patch('tools.llm_api.create_llm_client')
    def test_query_error(self, mock_create_client):
        self.mock_openai_client.chat.completions.create.side_effect = Exception("Test error")
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt")
        self.assertIsNone(response)

if __name__ == '__main__':
    unittest.main()
