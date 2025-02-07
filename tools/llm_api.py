#!/usr/bin/env python3

import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import sys
import base64
import mimetypes
import time
import json
from typing import Optional, Union, List, Dict, Any
from .token_tracker import TokenUsage, APIResponse, get_token_tracker
from .common.logging_config import setup_logging, LogLevel, LogFormat

logger = setup_logging(__name__)

class LLMApiError(Exception):
    """Custom exception for LLM API failures"""
    def __init__(self, message: str, provider: str, details: Optional[Dict[str, Any]] = None):
        self.provider = provider
        self.details = details or {}
        super().__init__(message)

class FileError(Exception):
    """Custom exception for file reading failures"""
    def __init__(self, message: str, file_path: str, details: Optional[Dict[str, Any]] = None):
        self.file_path = file_path
        self.details = details or {}
        super().__init__(message)

def load_environment() -> bool:
    """
    Load environment variables from .env files in order of precedence.
    
    Returns:
        bool: True if any environment file was loaded
        
    Note:
        Order of precedence:
        1. System environment variables (already loaded)
        2. .env.local (user-specific overrides)
        3. .env (project defaults)
        4. .env.example (example configuration)
    """
    env_files = ['.env.local', '.env', '.env.example']
    env_loaded = False
    
    logger.debug("Loading environment variables", extra={
        "context": {
            "working_directory": str(Path('.').absolute()),
            "env_files": env_files
        }
    })
    
    for env_file in env_files:
        env_path = Path('.') / env_file
        logger.debug("Checking environment file", extra={
            "context": {
                "file": str(env_path.absolute())
            }
        })
        if env_path.exists():
            logger.info("Loading environment file", extra={
                "context": {
                    "file": env_file
                }
            })
            load_dotenv(dotenv_path=env_path)
            env_loaded = True
            
            # Log loaded keys (but not values for security)
            with open(env_path) as f:
                keys = [line.split('=')[0].strip() for line in f if '=' in line and not line.startswith('#')]
                logger.debug("Loaded environment variables", extra={
                    "context": {
                        "file": env_file,
                        "keys": keys
                    }
                })
    
    if not env_loaded:
        logger.warning("No environment files found", extra={
            "context": {
                "system_env_keys": list(os.environ.keys())
            }
        })
    
    return env_loaded

def encode_image_file(image_path: str) -> tuple[str, str]:
    """
    Encode an image file to base64 and determine its MIME type.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (base64_encoded_string, mime_type)
        
    Raises:
        FileError: If the image file cannot be read or encoded, or if format is unsupported
    """
    path = Path(image_path)
    if not path.exists():
        logger.error("Image file not found", extra={
            "context": {
                "path": str(path)
            }
        })
        raise FileError("Image file not found", str(path))
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        logger.warning("Could not determine MIME type", extra={
            "context": {
                "path": str(path),
                "default_mime_type": "image/png"
            }
        })
        mime_type = 'image/png'
    elif mime_type not in ['image/png', 'image/jpeg']:
        logger.error("Unsupported image format", extra={
            "context": {
                "path": str(path),
                "mime_type": mime_type,
                "supported_formats": ['image/png', 'image/jpeg']
            }
        })
        raise FileError("Unsupported image format", str(path), {
            "mime_type": mime_type,
            "supported_formats": ['image/png', 'image/jpeg']
        })
    
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            logger.debug("Successfully encoded image", extra={
                "context": {
                    "path": str(path),
                    "encoded_size": len(encoded_string)
                }
            })
            return encoded_string, mime_type
    except Exception as e:
        logger.error("Failed to read/encode image file", extra={
            "context": {
                "path": str(path),
                "error": str(e)
            }
        })
        raise FileError("Failed to read/encode image file", str(path), {
            "error": str(e)
        })

def create_llm_client(provider: str = "openai") -> Any:
    """
    Create an LLM client with proper authentication.
    
    Args:
        provider: The API provider to use
        
    Returns:
        Any: Configured LLM client
        
    Raises:
        LLMApiError: If client creation fails or API key is missing
    """
    logger.debug("Creating LLM client", extra={
        "context": {
            "provider": provider
        }
    })
    
    try:
        if provider == "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("Missing API key", extra={
                    "context": {
                        "provider": provider,
                        "required_env": "OPENAI_API_KEY"
                    }
                })
                raise LLMApiError("OPENAI_API_KEY not found in environment variables", provider)
            return OpenAI(api_key=api_key)
            
        elif provider == "azure":
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            if not api_key:
                logger.error("Missing API key", extra={
                    "context": {
                        "provider": provider,
                        "required_env": "AZURE_OPENAI_API_KEY"
                    }
                })
                raise LLMApiError("AZURE_OPENAI_API_KEY not found in environment variables", provider)
            return AzureOpenAI(
                api_key=api_key,
                api_version="2024-08-01-preview",
                azure_endpoint="https://msopenai.openai.azure.com"
            )
            
        elif provider == "deepseek":
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                logger.error("Missing API key", extra={
                    "context": {
                        "provider": provider,
                        "required_env": "DEEPSEEK_API_KEY"
                    }
                })
                raise LLMApiError("DEEPSEEK_API_KEY not found in environment variables", provider)
            return OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
            )
            
        elif provider == "anthropic":
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.error("Missing API key", extra={
                    "context": {
                        "provider": provider,
                        "required_env": "ANTHROPIC_API_KEY"
                    }
                })
                raise LLMApiError("ANTHROPIC_API_KEY not found in environment variables", provider)
            return Anthropic(api_key=api_key)
            
        elif provider == "gemini":
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.error("Missing API key", extra={
                    "context": {
                        "provider": provider,
                        "required_env": "GOOGLE_API_KEY"
                    }
                })
                raise LLMApiError("GOOGLE_API_KEY not found in environment variables", provider)
            genai.configure(api_key=api_key)
            return genai
            
        elif provider == "local":
            return OpenAI(
                base_url="http://192.168.180.137:8006/v1",
                api_key="not-needed"
            )
            
        else:
            logger.error("Unsupported provider", extra={
                "context": {
                    "provider": provider,
                    "supported_providers": ["openai", "azure", "deepseek", "anthropic", "gemini", "local"]
                }
            })
            raise LLMApiError(f"Unsupported provider: {provider}", provider)
            
    except Exception as e:
        if isinstance(e, LLMApiError):
            raise
        logger.error("Failed to create LLM client", extra={
            "context": {
                "provider": provider,
                "error": str(e)
            }
        })
        raise LLMApiError(f"Failed to create {provider} client: {e}", provider)

def get_default_model(provider: str) -> str:
    """
    Get the default model name for a provider.
    
    Args:
        provider: The API provider
        
    Returns:
        str: Default model name
        
    Raises:
        LLMApiError: If provider is invalid
    """
    defaults = {
        "openai": "gpt-4o",
        "azure": os.getenv('AZURE_OPENAI_MODEL_DEPLOYMENT', 'gpt-4o-ms'),
        "deepseek": "deepseek-chat",
        "anthropic": "claude-3-sonnet-20240229",
        "gemini": "gemini-pro",
        "local": "Qwen/Qwen2.5-32B-Instruct-AWQ"
    }
    
    model = defaults.get(provider)
    if not model:
        logger.error("Invalid provider for default model", extra={
            "context": {
                "provider": provider,
                "supported_providers": list(defaults.keys())
            }
        })
        raise LLMApiError(f"Invalid provider: {provider}", provider)
    
    return model

def query_llm(
    prompt: str,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    provider: str = "openai",
    image_path: Optional[str] = None,
    system_content: Optional[str] = None
) -> Optional[str]:
    """
    Query an LLM with a prompt and optional image.
    
    Args:
        prompt: The prompt to send to the LLM
        client: Optional pre-configured LLM client
        model: Optional model name to use
        provider: LLM provider to use
        image_path: Optional path to an image file to include
        system_content: Optional system prompt
        
    Returns:
        str: LLM response
        
    Raises:
        LLMApiError: If there are issues with the LLM query
        FileError: If there are issues with image input or file reading
    """
    start_time = time.time()
    
    try:
        if not client:
            client = create_llm_client(provider)
        
        if not model:
            model = get_default_model(provider)
            
        # Check for image support
        if image_path:
            if provider not in ["openai", "gemini"]:
                raise FileError("Image input not supported", provider, {
                    "provider": provider,
                    "supported_providers": ["openai", "gemini"]
                })
            
            # Encode image if path provided
            base64_image, mime_type = encode_image_file(image_path)
            
        logger.info("Querying LLM", extra={
            "context": {
                "provider": provider,
                "model": model,
                "prompt_length": len(prompt),
                "has_image": image_path is not None,
                "has_system_content": system_content is not None
            }
        })
        
        if provider in ["openai", "local", "deepseek", "azure"]:
            messages = [{"role": "user", "content": []}]
            
            # Add system content if provided
            if system_content:
                messages[0]["content"].append({
                    "type": "text",
                    "text": system_content
                })
            
            # Add text content
            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })
            
            # Add image content if provided
            if image_path:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })
            
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
            }
            
            # Add o1-specific parameters
            if model == "o1":
                kwargs["response_format"] = {"type": "text"}
                kwargs["reasoning_effort"] = "low"
                del kwargs["temperature"]
            
            response = client.chat.completions.create(**kwargs)
            thinking_time = time.time() - start_time
            
            # Track token usage
            token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                reasoning_tokens=response.usage.completion_tokens_details.reasoning_tokens if hasattr(response.usage, 'completion_tokens_details') else None
            )
            
            # Calculate cost
            cost = get_token_tracker().calculate_openai_cost(
                token_usage.prompt_tokens,
                token_usage.completion_tokens,
                model
            )
            
            # Track the request
            api_response = APIResponse(
                content=response.choices[0].message.content,
                token_usage=token_usage,
                cost=cost,
                thinking_time=thinking_time,
                provider=provider,
                model=model
            )
            get_token_tracker().track_request(api_response)
            
            logger.info("LLM response received", extra={
                "context": {
                    "provider": provider,
                    "model": model,
                    "elapsed_seconds": thinking_time,
                    "response_length": len(response.choices[0].message.content),
                    "token_usage": {
                        "prompt": token_usage.prompt_tokens,
                        "completion": token_usage.completion_tokens,
                        "total": token_usage.total_tokens,
                        "reasoning": token_usage.reasoning_tokens
                    },
                    "cost": cost
                }
            })
            
            return response.choices[0].message.content
            
        elif provider == "anthropic":
            messages = [{"role": "user", "content": []}]
            
            # Add system content if provided
            if system_content:
                messages[0]["content"].append({
                    "type": "text",
                    "text": system_content
                })
            
            # Add text content
            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })
            
            # Add image content if provided
            if image_path:
                messages[0]["content"].append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image
                    }
                })
            
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=messages
            )
            thinking_time = time.time() - start_time
            
            # Track token usage
            token_usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
            
            # Calculate cost
            cost = get_token_tracker().calculate_claude_cost(
                token_usage.prompt_tokens,
                token_usage.completion_tokens,
                model
            )
            
            # Track the request
            api_response = APIResponse(
                content=response.content[0].text,
                token_usage=token_usage,
                cost=cost,
                thinking_time=thinking_time,
                provider=provider,
                model=model
            )
            get_token_tracker().track_request(api_response)
            
            logger.info("LLM response received", extra={
                "context": {
                    "provider": provider,
                    "model": model,
                    "elapsed_seconds": thinking_time,
                    "response_length": len(response.content[0].text),
                    "token_usage": {
                        "prompt": token_usage.prompt_tokens,
                        "completion": token_usage.completion_tokens,
                        "total": token_usage.total_tokens
                    },
                    "cost": cost
                }
            })
            
            return response.content[0].text
            
        elif provider == "gemini":
            model = client.GenerativeModel(model)
            response = model.generate_content(prompt)
            thinking_time = time.time() - start_time
            
            logger.info("LLM response received", extra={
                "context": {
                    "provider": provider,
                    "model": model,
                    "elapsed_seconds": thinking_time,
                    "response_length": len(response.text)
                }
            })
            
            return response.text
            
    except Exception as e:
        if isinstance(e, (LLMApiError, FileError)):
            raise
        logger.error("LLM query failed", extra={
            "context": {
                "provider": provider,
                "model": model,
                "error": str(e),
                "elapsed_seconds": time.time() - start_time
            }
        })
        raise LLMApiError(f"Failed to query {provider} LLM: {e}", provider, {
            "model": model,
            "error": str(e),
            "elapsed_seconds": time.time() - start_time
        })

def read_content_or_file(value: str) -> str:
    """
    Read content from a string or file if prefixed with @.
    
    Args:
        value: String value or @file path
        
    Returns:
        str: Content from string or file
        
    Raises:
        FileError: If file cannot be read
    """
    if not value:
        return ""
    
    if value.startswith('@'):
        file_path = value[1:]  # Remove @ prefix
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.debug("Read file content", extra={
                    "context": {
                        "path": file_path,
                        "content_length": len(content)
                    }
                })
                return content
        except Exception as e:
            logger.error("Failed to read file", extra={
                "context": {
                    "path": file_path,
                    "error": str(e)
                }
            })
            raise FileError("Failed to read file", file_path, {
                "error": str(e)
            })
    return value

def main():
    parser = argparse.ArgumentParser(
        description='Query an LLM with a prompt',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--prompt', type=str, help='The prompt to send to the LLM (prefix with @ to read from file)', required=True)
    parser.add_argument('--system', type=str, help='System prompt to use (prefix with @ to read from file)')
    parser.add_argument('--file', type=str, action='append', help='Path to a file to include in the prompt. Can be specified multiple times.')
    parser.add_argument('--provider',
                       choices=['openai', 'anthropic', 'gemini', 'local', 'deepseek', 'azure'],
                       default='openai',
                       help='The API provider to use')
    parser.add_argument('--model', type=str, help='The model to use (default depends on provider)')
    parser.add_argument('--image', type=str, help='Path to an image file to attach to the prompt')
    
    # Add output format options
    parser.add_argument('--format',
                       choices=['text', 'json', 'markdown'],
                       default='text',
                       help='Output format')
    
    # Add mutually exclusive logging options
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument('--log-level',
                          choices=['debug', 'info', 'warning', 'error', 'quiet'],
                          default='info',
                          help='Set the logging level')
    log_group.add_argument('--debug',
                          action='store_true',
                          help='Enable debug logging (equivalent to --log-level debug)')
    log_group.add_argument('--quiet',
                          action='store_true',
                          help='Minimize output (equivalent to --log-level quiet)')
    
    # Add log format option
    parser.add_argument('--log-format',
                       choices=['text', 'json', 'structured'],
                       default='text',
                       help='Log output format')
    
    args = parser.parse_args()

    # Configure logging
    log_level = LogLevel.DEBUG if args.debug else (
        LogLevel.QUIET if args.quiet else 
        LogLevel.from_string(args.log_level)
    )
    log_format = LogFormat(args.log_format)
    logger = setup_logging(__name__, level=log_level, format_type=log_format)
    logger.debug("Debug logging enabled", extra={
        "context": {
            "log_level": log_level.value,
            "log_format": log_format.value
        }
    })

    try:
        # Load environment variables
        load_environment()
        
        # Get default model if not specified
        if not args.model:
            args.model = get_default_model(args.provider)
            logger.debug("Using default model", extra={
                "context": {
                    "provider": args.provider,
                    "model": args.model
                }
            })

        # Read prompt and system content
        prompt = read_content_or_file(args.prompt)
        system_content = read_content_or_file(args.system) if args.system else ""
        
        # Read file contents if specified
        file_contents = []
        if args.file:
            for file_path in args.file:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_contents.append(f"\nFile: {file_path}\n{content}")
                        logger.debug("Read included file", extra={
                            "context": {
                                "path": file_path,
                                "content_length": len(content)
                            }
                        })
                except Exception as e:
                    logger.error("Failed to read included file", extra={
                        "context": {
                            "path": file_path,
                            "error": str(e)
                        }
                    })
                    raise FileError(f"Failed to read file {file_path}: {e}", file_path, {
                        "error": str(e)
                    })
        
        # Combine prompt with file contents
        if file_contents:
            prompt = prompt + "\n\nIncluded files:" + "\n---".join(file_contents)

        # Create client and query LLM
        client = create_llm_client(args.provider)
        response = query_llm(
            prompt,
            client,
            args.model,
            args.provider,
            args.image,
            system_content
        )
        
        if response:
            if args.format == 'json':
                print(json.dumps({
                    "response": response,
                    "model": args.model,
                    "provider": args.provider
                }, indent=2))
            elif args.format == 'markdown':
                print(f"# LLM Response\n\n{response}\n\n---\n*Model: {args.model} ({args.provider})*")
            else:
                print(response)
        else:
            logger.error("No response received", extra={
                "context": {
                    "provider": args.provider,
                    "model": args.model
                }
            })
            sys.exit(1)
            
    except FileError as e:
        logger.error(str(e), extra={
            "context": {
                "file_path": e.file_path,
                "details": e.details
            }
        })
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", extra={
            "context": {
                "error": str(e)
            }
        })
        sys.exit(1)

if __name__ == "__main__":
    main()