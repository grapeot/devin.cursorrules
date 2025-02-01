import os
import pytest
from tools.llm_api import load_environment

# Load environment at module level to ensure it's available for skip checks
load_environment()

# Example values from .env.example that indicate unconfigured keys
EXAMPLE_VALUES = {
    'OPENAI_API_KEY': 'your_openai_api_key_here',
    'ANTHROPIC_API_KEY': 'your_anthropic_api_key_here',
    'DEEPSEEK_API_KEY': 'your_deepseek_api_key_here',
    'GOOGLE_API_KEY': 'your_google_api_key_here',
    'AZURE_OPENAI_API_KEY': 'your_azure_openai_api_key_here',
    'AZURE_OPENAI_MODEL_DEPLOYMENT': 'gpt-4o-ms'
}

def get_skip_reason(env_var: str) -> str:
    """Get a descriptive reason why the test was skipped"""
    value = os.getenv(env_var, '').strip()
    if not value:
        return f"{env_var} is not set in environment"
    if value == EXAMPLE_VALUES.get(env_var, ''):
        return f"{env_var} is still set to example value: {value}"
    return f"{env_var} is not properly configured"

def is_unconfigured(env_var: str) -> bool:
    """Check if an environment variable is unset or set to its example value"""
    value = os.getenv(env_var, '').strip()
    return not value or value == EXAMPLE_VALUES.get(env_var, '')

def requires_openai(func):
    return pytest.mark.skipif(
        is_unconfigured('OPENAI_API_KEY'),
        reason=get_skip_reason('OPENAI_API_KEY')
    )(func)

def requires_anthropic(func):
    return pytest.mark.skipif(
        is_unconfigured('ANTHROPIC_API_KEY'),
        reason=get_skip_reason('ANTHROPIC_API_KEY')
    )(func)

def requires_azure(func):
    key_reason = get_skip_reason('AZURE_OPENAI_API_KEY')
    deploy_reason = get_skip_reason('AZURE_OPENAI_MODEL_DEPLOYMENT')
    return pytest.mark.skipif(
        is_unconfigured('AZURE_OPENAI_API_KEY') or is_unconfigured('AZURE_OPENAI_MODEL_DEPLOYMENT'),
        reason=f"Azure OpenAI not configured: {key_reason} and {deploy_reason}"
    )(func)

def requires_deepseek(func):
    return pytest.mark.skipif(
        is_unconfigured('DEEPSEEK_API_KEY'),
        reason=get_skip_reason('DEEPSEEK_API_KEY')
    )(func)

def requires_gemini(func):
    return pytest.mark.skipif(
        is_unconfigured('GOOGLE_API_KEY'),
        reason=get_skip_reason('GOOGLE_API_KEY')
    )(func)

def requires_openai_o1(func):
    return pytest.mark.skipif(
        is_unconfigured('OPENAI_API_KEY'),
        reason=get_skip_reason('OPENAI_API_KEY')
    )(func) 