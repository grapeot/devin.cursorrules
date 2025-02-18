import unittest
from unittest.mock import patch, MagicMock, mock_open
from tools.llm_api import create_llm_client, query_llm, load_environment
import os
import google.generativeai as genai
import io
import sys

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
        self.mock_gemini_client = MagicMock()
        
        # Set up OpenAI-style response
        self.mock_openai_response = MagicMock()
        self.mock_openai_choice = MagicMock()
        self.mock_openai_message = MagicMock()
        self.mock_openai_message.content = "Test OpenAI response"
        self.mock_openai_choice.message = self.mock_openai_message
        self.mock_openai_response.choices = [self.mock_openai_choice]
        self.mock_openai_client.chat.completions.create.return_value = self.mock_openai_response
        
        # Set up Anthropic-style response
        self.mock_anthropic_response = MagicMock()
        self.mock_anthropic_content = MagicMock()
        self.mock_anthropic_content.text = "Test Anthropic response"
        self.mock_anthropic_response.content = [self.mock_anthropic_content]
        self.mock_anthropic_client.messages.create.return_value = self.mock_anthropic_response
        
        # Set up Gemini-style response
        self.mock_gemini_model = MagicMock()
        self.mock_gemini_response = MagicMock()
        self.mock_gemini_response.text = "Test Gemini response"
        self.mock_gemini_model.generate_content.return_value = self.mock_gemini_response
        self.mock_gemini_client.GenerativeModel.return_value = self.mock_gemini_model
        
        # Set up SiliconFlow-style response
        self.mock_siliconflow_response = MagicMock()
        self.mock_siliconflow_choice = MagicMock()
        self.mock_siliconflow_message = MagicMock()
        self.mock_siliconflow_message.content = "Test Siliconflow response"
        self.mock_siliconflow_choice.message = self.mock_siliconflow_message
        self.mock_siliconflow_response.choices = [self.mock_siliconflow_choice]
        
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-openai-key',
            'DEEPSEEK_API_KEY': 'test-deepseek-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'GOOGLE_API_KEY': 'test-google-key',
            'AZURE_OPENAI_API_KEY': 'test-azure-key',
            'AZURE_OPENAI_MODEL_DEPLOYMENT': 'test-model-deployment',
            'SILICONFLOW_API_KEY': 'test-siliconflow-key'
        })
        self.env_patcher.start()
        
        # Set up Azure OpenAI mock
        self.mock_azure_response = MagicMock()
        self.mock_azure_choice = MagicMock()
        self.mock_azure_message = MagicMock()
        self.mock_azure_message.content = "Test Azure OpenAI response"
        self.mock_azure_choice.message = self.mock_azure_message
        self.mock_azure_response.choices = [self.mock_azure_choice]
        self.mock_azure_client = MagicMock()
        self.mock_azure_client.chat.completions.create.return_value = self.mock_azure_response

    def tearDown(self):
        self.env_patcher.stop()

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.OpenAI')
    def test_create_openai_client(self, mock_openai):
        mock_openai.return_value = self.mock_openai_client
        client = create_llm_client("openai")
        mock_openai.assert_called_once_with(api_key='test-openai-key')
        self.assertEqual(client, self.mock_openai_client)

    @unittest.skipIf(skip_llm_tests, skip_message)
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

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.OpenAI')
    def test_create_deepseek_client(self, mock_openai):
        mock_openai.return_value = self.mock_openai_client
        client = create_llm_client("deepseek")
        mock_openai.assert_called_once_with(
            api_key='test-deepseek-key',
            base_url="https://api.deepseek.com/v1"
        )
        self.assertEqual(client, self.mock_openai_client)

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.OpenAI')
    def test_create_siliconflow_client(self, mock_openai):
        mock_openai.return_value = self.mock_openai_client
        client = create_llm_client("siliconflow")
        mock_openai.assert_called_once_with(
            api_key='test-siliconflow-key',
            base_url="https://api.siliconflow.cn/v1"
        )
        self.assertEqual(client, self.mock_openai_client)

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.Anthropic')
    def test_create_anthropic_client(self, mock_anthropic):
        mock_anthropic.return_value = self.mock_anthropic_client
        client = create_llm_client("anthropic")
        mock_anthropic.assert_called_once_with(api_key='test-anthropic-key')
        self.assertEqual(client, self.mock_anthropic_client)

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.genai')
    def test_create_gemini_client(self, mock_genai):
        client = create_llm_client("gemini")
        mock_genai.configure.assert_called_once_with(api_key='test-google-key')
        self.assertEqual(client, mock_genai)

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.OpenAI')
    def test_create_local_client(self, mock_openai):
        mock_openai.return_value = self.mock_openai_client
        client = create_llm_client("local")
        mock_openai.assert_called_once_with(
            base_url="http://192.168.180.137:8006/v1",
            api_key="not-needed"
        )
        self.assertEqual(client, self.mock_openai_client)

    @unittest.skipIf(skip_llm_tests, skip_message)
    def test_create_invalid_provider(self):
        with self.assertRaises(ValueError):
            create_llm_client("invalid_provider")

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_openai(self, mock_create_client):
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", provider="openai")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_azure(self, mock_create_client):
        mock_create_client.return_value = self.mock_azure_client
        response = query_llm("Test prompt", provider="azure")
        self.assertEqual(response, "Test Azure OpenAI response")
        self.mock_azure_client.chat.completions.create.assert_called_once_with(
            model=os.getenv('AZURE_OPENAI_MODEL_DEPLOYMENT', 'gpt-4o-ms'),
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_deepseek(self, mock_create_client):
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", provider="deepseek")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_siliconflow(self, mock_create_client):
        self.mock_openai_client.chat.completions.create.return_value = self.mock_siliconflow_response
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", provider="siliconflow")
        self.assertEqual(response, "Test Siliconflow response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_anthropic(self, mock_create_client):
        mock_create_client.return_value = self.mock_anthropic_client
        response = query_llm("Test prompt", provider="anthropic")
        self.assertEqual(response, "Test Anthropic response")
        self.mock_anthropic_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}]
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_gemini(self, mock_create_client):
        mock_create_client.return_value = self.mock_gemini_client
        response = query_llm("Test prompt", provider="gemini")
        self.assertEqual(response, "Test Gemini response")
        self.mock_gemini_client.GenerativeModel.assert_called_once_with("gemini-pro")
        self.mock_gemini_model.generate_content.assert_called_once_with("Test prompt")

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_local(self, mock_create_client):
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", provider="local")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_with_custom_model(self, mock_create_client):
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", model="custom-model")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="custom-model",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_o1_model(self, mock_create_client):
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", model="o1")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="o1",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            response_format={"type": "text"},
            reasoning_effort="low"
        )

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_with_existing_client(self, mock_create_client):
        response = query_llm("Test prompt", client=self.mock_openai_client)
        self.assertEqual(response, "Test OpenAI response")
        mock_create_client.assert_not_called()

    @unittest.skipIf(skip_llm_tests, skip_message)
    @patch('tools.llm_api.create_llm_client')
    def test_query_error(self, mock_create_client):
        self.mock_openai_client.chat.completions.create.side_effect = Exception("Test error")
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt")
        self.assertIsNone(response)

if __name__ == '__main__':
    unittest.main()
