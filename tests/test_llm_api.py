import unittest
import pytest
from unittest.mock import patch, MagicMock, mock_open
from tools.llm_api import (
    create_llm_client,
    query_llm,
    load_environment,
    LLMApiError,
    encode_image_file,
    get_default_model,
    read_content_or_file
)
from tools.token_tracker import TokenUsage, APIResponse
import os
import google.generativeai as genai
import io
import sys
import base64
import mimetypes
import tempfile
from tools.common.errors import FileError, APIError

class AsyncContextManagerMock:
    def __init__(self, response):
        self.response = response
    
    async def __aenter__(self):
        return self.response
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

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
        with self.assertRaises(LLMApiError) as cm:
            create_llm_client("invalid_provider")
        self.assertIn("Unsupported provider: invalid_provider", str(cm.exception))

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
    def test_query_deepseek(self, mock_create_client):
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", provider="deepseek", model="deepseek-chat")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            temperature=0.7
        )

    @patch('tools.llm_api.create_llm_client')
    def test_query_anthropic(self, mock_create_client):
        mock_create_client.return_value = self.mock_anthropic_client
        response = query_llm("Test prompt", provider="anthropic", model="claude-3-5-sonnet-20241022")
        self.assertEqual(response, "Test Anthropic response")
        self.mock_anthropic_client.messages.create.assert_called_once_with(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}]
        )

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
    def test_query_o1_model(self, mock_create_client):
        mock_create_client.return_value = self.mock_openai_client
        response = query_llm("Test prompt", provider="openai", model="o1")
        self.assertEqual(response, "Test OpenAI response")
        self.mock_openai_client.chat.completions.create.assert_called_once_with(
            model="o1",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
            response_format={"type": "text"},
            reasoning_effort="low"
        )

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
        with self.assertRaises(LLMApiError) as cm:
            query_llm("Test prompt")
        self.assertIn("Failed to query openai LLM: Test error", str(cm.exception))

    def test_encode_image_file(self):
        """Test image file encoding"""
        # Create a test image file
        test_image = b'fake image data'
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(test_image)
            test_file = f.name
        
        try:
            # Test PNG image
            with patch('mimetypes.guess_type', return_value=('image/png', None)):
                base64_data, mime_type = encode_image_file(test_file)
                self.assertEqual(mime_type, 'image/png')
                self.assertEqual(base64.b64decode(base64_data), test_image)
            
            # Test JPEG image
            with patch('mimetypes.guess_type', return_value=('image/jpeg', None)):
                base64_data, mime_type = encode_image_file(test_file)
                self.assertEqual(mime_type, 'image/jpeg')
                self.assertEqual(base64.b64decode(base64_data), test_image)
            
            # Test unknown type defaulting to PNG
            with patch('mimetypes.guess_type', return_value=(None, None)):
                base64_data, mime_type = encode_image_file(test_file)
                self.assertEqual(mime_type, 'image/png')
                self.assertEqual(base64.b64decode(base64_data), test_image)
        finally:
            os.unlink(test_file)

    def test_get_default_model(self):
        """Test default model selection"""
        # Test OpenAI default
        self.assertEqual(get_default_model("openai"), "gpt-4o")
        
        # Test Azure default
        self.assertEqual(get_default_model("azure"), "test-model-deployment")
        
        # Test Anthropic default
        self.assertEqual(get_default_model("anthropic"), "claude-3-sonnet-20240229")
        
        # Test Gemini default
        self.assertEqual(get_default_model("gemini"), "gemini-pro")
        
        # Test Deepseek default
        self.assertEqual(get_default_model("deepseek"), "deepseek-chat")
        
        # Test invalid provider
        with self.assertRaises(LLMApiError) as cm:
            get_default_model("invalid")
        self.assertIn("Invalid provider", str(cm.exception))

    def test_query_llm_with_image(self):
        """Test LLM querying with image input"""
        # Create a test image file
        test_image = b'fake image data'
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(test_image)
            test_file = f.name
        
        try:
            # Mock image encoding
            with patch('tools.llm_api.encode_image_file') as mock_encode:
                mock_encode.return_value = ('base64data', 'image/png')
                
                # Test OpenAI vision model
                response = query_llm(
                    "Describe this image",
                    client=self.mock_openai_client,
                    provider="openai",
                    model="gpt-4-vision-preview",
                    image_path=test_file
                )
                self.assertEqual(response, "Test OpenAI response")
                
                # Verify image was included in messages
                calls = self.mock_openai_client.chat.completions.create.call_args_list
                messages = calls[0][1]['messages']
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0]['role'], 'user')
                self.assertEqual(len(messages[0]['content']), 2)
                self.assertEqual(messages[0]['content'][0]['type'], 'text')
                self.assertEqual(messages[0]['content'][1]['type'], 'image_url')
                
                # Test Gemini vision model
                response = query_llm(
                    "Describe this image",
                    client=self.mock_gemini_client,
                    provider="gemini",
                    model="gemini-pro-vision",
                    image_path=test_file
                )
                self.assertEqual(response, "Test Gemini response")
        finally:
            os.unlink(test_file)

    def test_provider_specific_errors(self):
        """Test provider-specific error handling"""
        # Test OpenAI rate limit error
        self.mock_openai_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        with self.assertRaises(LLMApiError) as cm:
            query_llm("Test prompt", client=self.mock_openai_client, provider="openai")
        self.assertIn("Rate limit exceeded", str(cm.exception))
        
        # Test Anthropic timeout error
        self.mock_anthropic_client.messages.create.side_effect = Exception("Request timed out")
        with self.assertRaises(LLMApiError) as cm:
            query_llm("Test prompt", client=self.mock_anthropic_client, provider="anthropic")
        self.assertIn("Request timed out", str(cm.exception))
        
        # Test Gemini API error
        self.mock_gemini_model.generate_content.side_effect = Exception("API error")
        with self.assertRaises(LLMApiError) as cm:
            query_llm("Test prompt", client=self.mock_gemini_client, provider="gemini")
        self.assertIn("API error", str(cm.exception))

    def test_read_content_or_file(self):
        """Test content/file reading"""
        # Test direct content
        content = read_content_or_file("Test content")
        self.assertEqual(content, "Test content")
        
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('File content')
            test_file = f.name
        
        try:
            # Test file content
            content = read_content_or_file(f"@{test_file}")
            self.assertEqual(content, "File content")
            
            # Test empty input
            content = read_content_or_file("")
            self.assertEqual(content, "")
        finally:
            os.unlink(test_file)

def test_create_invalid_provider():
    with pytest.raises(LLMApiError) as exc_info:
        create_llm_client("invalid_provider")
    assert "Unsupported provider: invalid_provider" in str(exc_info.value)

def test_create_llm_client_missing_api_key():
    # Clear the environment variable
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    with pytest.raises(LLMApiError) as exc_info:
        create_llm_client("openai")
    assert "OPENAI_API_KEY not found in environment variables" in str(exc_info.value)

if __name__ == '__main__':
    unittest.main()
