#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import sys
import time
import json
import aiohttp
from typing import Optional, Dict, Any, Union, List
from .token_tracker import TokenUsage, APIResponse, get_token_tracker
from .common.logging_config import setup_logging, LogLevel, LogFormat
from .common.errors import ToolError, ValidationError, APIError, FileError
from .common.formatting import format_output, format_duration, format_cost
from .common.cli import create_parser, get_log_config

logger = setup_logging(__name__)

STATUS_FILE = '.cursorrules'

def validate_file_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate a file path.
    
    Args:
        path: Path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Path: Validated Path object
        
    Raises:
        FileError: If path is invalid or file doesn't exist when required
    """
    try:
        path_obj = Path(path)
        if must_exist and not path_obj.exists():
            logger.error("File not found", extra={
                "context": {
                    "path": str(path_obj),
                    "must_exist": must_exist
                }
            })
            raise FileError("File not found", str(path_obj))
        if path_obj.exists() and not path_obj.is_file():
            logger.error("Path exists but is not a file", extra={
                "context": {
                    "path": str(path_obj)
                }
            })
            raise FileError("Not a file", str(path_obj))
        return path_obj
    except Exception as e:
        if isinstance(e, FileError):
            raise
        logger.error("Invalid file path", extra={
            "context": {
                "path": str(path),
                "error": str(e)
            }
        })
        raise FileError("Invalid file path", str(path), {
            "error": str(e)
        })

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

def read_plan_status() -> str:
    """
    Read the content of the plan status file, only including content after Multi-Agent Scratchpad.
    
    Returns:
        str: Content of the Multi-Agent Scratchpad section
        
    Raises:
        FileError: If there are issues reading the file
        ValidationError: If section not found
    """
    try:
        status_file = validate_file_path(STATUS_FILE)
        logger.debug("Reading status file", extra={
            "context": {
                "file": str(status_file)
            }
        })
        
        with open(status_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Find the Multi-Agent Scratchpad section
            scratchpad_marker = "# Multi-Agent Scratchpad"
            if scratchpad_marker not in content:
                logger.error("Multi-Agent Scratchpad section not found", extra={
                    "context": {
                        "file": str(status_file),
                        "marker": scratchpad_marker
                    }
                })
                raise ValidationError(
                    f"'{scratchpad_marker}' section not found in status file",
                    {
                        "file": str(status_file),
                        "marker": scratchpad_marker
                    }
                )
            
            section_content = content[content.index(scratchpad_marker) + len(scratchpad_marker):]
            if not section_content.strip():
                logger.error("Empty Multi-Agent Scratchpad section", extra={
                    "context": {
                        "file": str(status_file)
                    }
                })
                raise ValidationError(
                    "Multi-Agent Scratchpad section is empty",
                    {
                        "file": str(status_file)
                    }
                )
            
            logger.debug("Found Multi-Agent Scratchpad section", extra={
                "context": {
                    "file": str(status_file),
                    "content_length": len(section_content)
                }
            })
            return section_content
                
    except Exception as e:
        if isinstance(e, (FileError, ValidationError)):
            raise
        logger.error("Failed to read status file", extra={
            "context": {
                "file": STATUS_FILE,
                "error": str(e)
            }
        })
        raise FileError("Failed to read status file", STATUS_FILE, {
            "error": str(e)
        })

def read_file_content(file_path: str) -> str:
    """
    Read content from a specified file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        str: File content
        
    Raises:
        FileError: If there are issues reading the file
    """
    try:
        path = validate_file_path(file_path)
        logger.debug("Reading file", extra={
            "context": {
                "path": str(path)
            }
        })
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                logger.warning("File is empty", extra={
                    "context": {
                        "path": str(path)
                    }
                })
            else:
                logger.debug("Successfully read file", extra={
                    "context": {
                        "path": str(path),
                        "content_length": len(content)
                    }
                })
            return content
            
    except Exception as e:
        if isinstance(e, FileError):
            raise
        logger.error("Failed to read file", extra={
            "context": {
                "path": file_path,
                "error": str(e)
            }
        })
        raise FileError("Failed to read file", file_path, {
            "error": str(e)
        })

def create_llm_client() -> OpenAI:
    """
    Create OpenAI client with proper authentication.
    
    Returns:
        OpenAI: Configured OpenAI client
        
    Raises:
        PlanExecError: If API key is missing or invalid
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("Missing API key", extra={
            "context": {
                "required_env": "OPENAI_API_KEY"
            }
        })
        raise FileError(
            "OPENAI_API_KEY not found in environment variables",
            "client_creation"
        )
    
    try:
        logger.debug("Creating OpenAI client")
        return OpenAI(api_key=api_key)
    except Exception as e:
        logger.error("Failed to create OpenAI client", extra={
            "context": {
                "error": str(e)
            }
        })
        raise FileError(
            f"Failed to create OpenAI client: {e}",
            "client_creation"
        )

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
            raise FileError("Failed to read file", file_path, {
                "error": str(e)
            })
    return value

def query_llm(plan_content: str, user_prompt: Optional[str] = None, file_content: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
    """
    Query the LLM with combined prompts.
    
    Args:
        plan_content: Current plan and status content
        user_prompt: Optional additional user prompt
        file_content: Optional content from a specific file
        system_prompt: Optional system prompt to override default
        
    Returns:
        str: LLM response
        
    Raises:
        ValidationError: If inputs are invalid
        APIError: If there are issues with the LLM query
    """
    if not plan_content or not plan_content.strip():
        logger.error("Empty plan content", extra={
            "context": {
                "has_user_prompt": user_prompt is not None,
                "has_file_content": file_content is not None
            }
        })
        raise ValidationError("Plan content cannot be empty", {
            "has_user_prompt": user_prompt is not None,
            "has_file_content": file_content is not None
        })
    
    try:
        client = create_llm_client()
        
        # Initialize system prompt
        system_prompt = system_prompt or """You are working on a multi-agent context. The executor is the one who actually does the work. And you are the planner. Now the executor is asking you for help. Please analyze the provided project plan and status, then address the executor's specific query or request.

You need to think like a founder. Prioritize agility and don't over-engineer. Think deep. Try to foresee challenges and derisk earlier. If opportunity sizing or probing experiments can reduce risk with low cost, instruct the executor to do them."""
        
        # Combine prompts with exact text preserved
        combined_prompt = f"""Project Plan and Status:
======
{plan_content}
======
"""
        
        if file_content:
            combined_prompt += f"\nFile Content:\n======\n{file_content}\n======\n"
        
        if user_prompt:
            combined_prompt += f"\nUser Query:\n{user_prompt}\n"
        
        combined_prompt += """\nYour response should be focusing on revising the Multi-Agent Scratchpad section in the .cursorrules file. There is no need to regenerate the entire document. You can use the following format to prompt how to revise the document:

<<<<<<<SEARCH
<text in the original document>
=======
<Proprosed changes>
>>>>>>>

We will do the actual changes in the .cursorrules file."""
        
        logger.info("Querying LLM", extra={
            "context": {
                "prompt_length": len(combined_prompt),
                "has_user_prompt": user_prompt is not None,
                "has_file_content": file_content is not None,
                "has_system_prompt": system_prompt is not None
            }
        })
        
        start_time = time.time()
        response = client.chat.completions.create(
            model="o1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_prompt}
            ],
            response_format={"type": "text"},
            reasoning_effort="low"
        )
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
            "o1"
        )
        
        # Track the request
        api_response = APIResponse(
            content=response.choices[0].message.content,
            token_usage=token_usage,
            cost=cost,
            thinking_time=thinking_time,
            provider="openai",
            model="o1"
        )
        get_token_tracker().track_request(api_response)
        
        logger.info("LLM response received", extra={
            "context": {
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
        
    except Exception as e:
        if isinstance(e, (ValidationError, APIError)):
            raise
        logger.error("Failed to query LLM", extra={
            "context": {
                "error": str(e),
                "elapsed_seconds": time.time() - start_time
            }
        })
        raise APIError("Failed to query LLM", "openai", {
            "error": str(e),
            "elapsed_seconds": time.time() - start_time
        })

def format_output(response: str, format_type: str = 'text') -> str:
    """
    Format the LLM response for display.
    
    Args:
        response: Raw LLM response
        format_type: Output format (text, json, or markdown)
        
    Returns:
        str: Formatted output string
        
    Raises:
        PlanExecError: If response is invalid
        ValidationError: If format_type is invalid
    """
    if not response or not response.strip():
        logger.error("Empty LLM response", extra={
            "context": {
                "format_type": format_type
            }
        })
        raise FileError("Empty response from LLM", "output_formatting")
    
    if format_type == 'json':
        return json.dumps({
            "response": response,
            "model": "o1",
            "provider": "openai",
            "context": "plan_exec"
        }, indent=2)
    elif format_type == 'markdown':
        return f"""# Plan Execution Response

{response}

---
*Model: o1 (OpenAI)*
*Context: Plan Execution*"""
    elif format_type == 'text':
        # Add clear section markers for text format
        sections = [
            'Following is the instruction on how to revise the Multi-Agent Scratchpad section in .cursorrules:',
            '=' * 72,
            response,
            '=' * 72,
            'End of instruction'
        ]
        return '\n'.join(sections)
    else:
        logger.error("Invalid format type", extra={
            "context": {
                "format_type": format_type,
                "valid_formats": ["text", "json", "markdown"]
            }
        })
        raise ValidationError("Invalid format type", {
            "format_type": format_type,
            "valid_formats": ["text", "json", "markdown"]
        })

def validate_plan(plan: Dict[str, Any]) -> None:
    """
    Validate plan structure.
    
    Args:
        plan: Plan dictionary to validate
        
    Raises:
        ValidationError: If plan structure is invalid
    """
    required_keys = ["goal", "steps"]
    missing_keys = [k for k in required_keys if k not in plan]
    if missing_keys:
        raise ValidationError("Missing required plan keys", {
            "missing_keys": missing_keys,
            "plan": plan
        })
    
    if not isinstance(plan["steps"], list):
        raise ValidationError("Steps must be a list", {
            "steps_type": type(plan["steps"]).__name__
        })
    
    for i, step in enumerate(plan["steps"]):
        if not isinstance(step, dict):
            raise ValidationError(f"Step {i} must be a dictionary", {
                "step_type": type(step).__name__,
                "step_index": i
            })
        
        step_keys = ["description", "action", "expected_result"]
        missing_step_keys = [k for k in step_keys if k not in step]
        if missing_step_keys:
            raise ValidationError(f"Step {i} missing required keys", {
                "missing_keys": missing_step_keys,
                "step_index": i,
                "step": step
            })

def validate_execution_result(result: Dict[str, Any]) -> None:
    """
    Validate execution result structure.
    
    Args:
        result: Result dictionary to validate
        
    Raises:
        ValidationError: If result structure is invalid
    """
    required_keys = ["success", "output"]
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        raise ValidationError("Missing required result keys", {
            "missing_keys": missing_keys,
            "result": result
        })
    
    if not isinstance(result["success"], bool):
        raise ValidationError("Success must be a boolean", {
            "success_type": type(result["success"]).__name__
        })

async def execute_plan(
    plan: Dict[str, Any],
    session: Optional[aiohttp.ClientSession] = None,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Execute a plan using LLM for validation.
    
    Args:
        plan: Plan dictionary with goal and steps
        session: Optional aiohttp session to reuse
        timeout: Request timeout in seconds
        
    Returns:
        Dict containing execution results
        
    Raises:
        ValidationError: If plan structure is invalid
        PlanExecError: If execution fails
    """
    # Validate plan structure
    validate_plan(plan)
    
    logger.info("Starting plan execution", extra={
        "context": {
            "goal": plan["goal"],
            "num_steps": len(plan["steps"]),
            "timeout": timeout
        }
    })
    
    start_time = time.time()
    results = []
    total_cost = 0.0
    
    try:
        for i, step in enumerate(plan["steps"]):
            step_start = time.time()
            logger.info(f"Executing step {i+1}", extra={
                "context": {
                    "step_index": i,
                    "description": step["description"],
                    "action": step["action"]
                }
            })
            
            # Execute step action
            try:
                # TODO: Implement actual step execution
                # This is a placeholder that always succeeds
                step_result = {
                    "success": True,
                    "output": f"Executed: {step['action']}"
                }
                
                # Validate result structure
                validate_execution_result(step_result)
                
                step_elapsed = time.time() - step_start
                results.append({
                    "step_index": i,
                    "description": step["description"],
                    "action": step["action"],
                    "expected_result": step["expected_result"],
                    "success": step_result["success"],
                    "output": step_result["output"],
                    "elapsed_time": step_elapsed
                })
                
                logger.info(f"Step {i+1} completed", extra={
                    "context": {
                        "step_index": i,
                        "success": step_result["success"],
                        "elapsed_seconds": step_elapsed
                    }
                })
                
            except Exception as e:
                step_elapsed = time.time() - step_start
                logger.error(f"Step {i+1} failed", extra={
                    "context": {
                        "step_index": i,
                        "error": str(e),
                        "elapsed_seconds": step_elapsed
                    }
                })
                
                results.append({
                    "step_index": i,
                    "description": step["description"],
                    "action": step["action"],
                    "expected_result": step["expected_result"],
                    "success": False,
                    "error": str(e),
                    "elapsed_time": step_elapsed
                })
                
                raise FileError(
                    f"Step {i+1} failed: {e}",
                    "execution",
                    {
                        "step_index": i,
                        "description": step["description"],
                        "error": str(e)
                    }
                )
    
    except Exception as e:
        if not isinstance(e, (ValidationError, FileError)):
            logger.error("Unexpected error during execution", extra={
                "context": {
                    "error": str(e),
                    "elapsed_seconds": time.time() - start_time
                }
            })
            raise FileError(
                "Execution failed",
                "unknown",
                {
                    "error": str(e),
                    "elapsed_seconds": time.time() - start_time
                }
            )
        raise
    
    elapsed = time.time() - start_time
    
    result = {
        "goal": plan["goal"],
        "steps": results,
        "total_steps": len(results),
        "successful_steps": sum(1 for r in results if r["success"]),
        "failed_steps": sum(1 for r in results if not r["success"]),
        "elapsed_time": elapsed
    }
    
    logger.info("Plan execution completed", extra={
        "context": {
            "goal": plan["goal"],
            "total_steps": result["total_steps"],
            "successful_steps": result["successful_steps"],
            "failed_steps": result["failed_steps"],
            "elapsed_seconds": elapsed
        }
    })
    
    return result

def main():
    parser = create_parser('Execute a plan using LLM for validation')
    parser.add_argument('--plan', type=str, required=True,
                       help='JSON string or @file containing the plan')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds')
    
    args = parser.parse_args()
    
    # Configure logging
    log_config = get_log_config(args)
    logger = setup_logging(__name__, **log_config)
    logger.debug("Debug logging enabled", extra={
        "context": {
            "log_level": log_config["level"].value,
            "log_format": log_config["format_type"].value
        }
    })
    
    try:
        # Load plan
        if args.plan.startswith('@'):
            try:
                with open(args.plan[1:], 'r') as f:
                    plan = json.load(f)
            except Exception as e:
                raise ValidationError("Failed to load plan file", {
                    "file": args.plan[1:],
                    "error": str(e)
                })
        else:
            try:
                plan = json.loads(args.plan)
            except json.JSONDecodeError as e:
                raise ValidationError("Invalid plan JSON", {
                    "error": str(e),
                    "plan": args.plan
                })
        
        start_time = time.time()
        result = asyncio.run(execute_plan(
            plan,
            timeout=args.timeout
        ))
        elapsed = time.time() - start_time
        
        metadata = {
            "total_steps": result["total_steps"],
            "successful_steps": result["successful_steps"],
            "failed_steps": result["failed_steps"],
            "elapsed_time": format_duration(elapsed)
        }
        
        print(format_output(result, args.format, "Plan Execution Results", metadata))
        
        # Exit with error if any steps failed
        if result["failed_steps"] > 0:
            sys.exit(1)
            
    except ValidationError as e:
        logger.error("Invalid input", extra={"context": e.context})
        sys.exit(1)
    except FileError as e:
        logger.error("Execution error", extra={"context": e.context})
        sys.exit(1)
    except Exception as e:
        logger.error("Processing failed", extra={
            "context": {
                "error": str(e)
            }
        })
        sys.exit(1)

if __name__ == "__main__":
    main() 