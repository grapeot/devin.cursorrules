#!/usr/bin/env python3

import os
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
from pathlib import Path
import uuid
import sys
import tempfile
import shutil
from tabulate import tabulate
from datetime import datetime
from .common.logging_config import setup_logging, LogLevel, LogFormat
from .common.errors import ToolError, ValidationError, FileError
from .common.formatting import format_output, format_cost, format_duration

logger = setup_logging(__name__)

@dataclass
class TokenUsage:
    """Token usage information for an API request"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None

    def __post_init__(self):
        """Validate token counts"""
        if self.prompt_tokens < 0:
            raise ValidationError("prompt_tokens must be non-negative", {
                "field": "prompt_tokens",
                "value": self.prompt_tokens
            })
        if self.completion_tokens < 0:
            raise ValidationError("completion_tokens must be non-negative", {
                "field": "completion_tokens",
                "value": self.completion_tokens
            })
        if self.total_tokens < 0:
            raise ValidationError("total_tokens must be non-negative", {
                "field": "total_tokens",
                "value": self.total_tokens
            })
        if self.reasoning_tokens is not None and self.reasoning_tokens < 0:
            raise ValidationError("reasoning_tokens must be non-negative", {
                "field": "reasoning_tokens",
                "value": self.reasoning_tokens
            })

@dataclass
class APIResponse:
    """API response information"""
    content: str
    token_usage: TokenUsage
    cost: float
    thinking_time: float = 0.0
    provider: str = "openai"
    model: str = "unknown"

    def __post_init__(self):
        """Validate response data"""
        if not self.content:
            raise ValidationError("content cannot be empty")
        if self.cost < 0:
            raise ValidationError("cost must be non-negative", {
                "cost": self.cost
            })
        if self.thinking_time < 0:
            raise ValidationError("thinking_time must be non-negative", {
                "thinking_time": self.thinking_time
            })
        if not self.provider:
            raise ValidationError("provider cannot be empty")
        if not self.model:
            raise ValidationError("model cannot be empty")

class TokenTracker:
    """Track token usage and costs for API requests"""
    
    def __init__(self, session_id: Optional[str] = None, logs_dir: Optional[Path] = None):
        """
        Initialize token tracker.
        
        Args:
            session_id: Optional session identifier
            logs_dir: Optional directory for log files
        """
        self._session_id = session_id or str(int(time.time()))
        self._logs_dir = logs_dir or Path.home() / '.cursorrules' / 'logs'
        self._requests: List[Dict[str, Any]] = []
        self._session_start = time.time()
        
        logger.debug("Initializing token tracker", extra={
            "context": {
                "session_id": self._session_id,
                "logs_dir": str(self._logs_dir)
            }
        })
        
        try:
            # Create logs directory if it doesn't exist
            self._logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize session file
            self._session_file = self._logs_dir / f"session_{self._session_id}.json"
            
            # Only load existing session if it matches our session ID
            if self._session_file.exists() and session_id:
                logger.debug("Loading existing session", extra={
                    "context": {
                        "session_file": str(self._session_file)
                    }
                })
                session_data = load_session(self._session_file)
                if session_data and session_data.get('session_id') == self._session_id:
                    self._requests = session_data.get('requests', [])
                    self._session_start = session_data.get('start_time', self._session_start)
                    logger.info("Loaded existing session", extra={
                        "context": {
                            "request_count": len(self._requests)
                        }
                    })
        except Exception as e:
            logger.error("Failed to initialize token tracker", extra={
                "context": {
                    "session_id": self._session_id,
                    "logs_dir": str(self._logs_dir),
                    "error": str(e)
                }
            })
            raise
    
    def _save_session(self):
        """Save current session data to file"""
        try:
            session_data = {
                "session_id": self._session_id,
                "start_time": self._session_start,
                "requests": self._requests,
                "summary": self.get_session_summary()
            }
            
            with open(self._session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            logger.debug("Saved session data", extra={
                "context": {
                    "session_file": str(self._session_file),
                    "request_count": len(self._requests)
                }
            })
        except Exception as e:
            logger.error("Failed to save session data", extra={
                "context": {
                    "session_file": str(self._session_file),
                    "error": str(e)
                }
            })
            raise
    
    @property
    def logs_dir(self) -> Path:
        """Get the logs directory path"""
        return self._logs_dir
    
    @logs_dir.setter
    def logs_dir(self, path: Path):
        """
        Set the logs directory path and update session file path.
        
        Args:
            path: New logs directory path
            
        Raises:
            ValueError: If path is invalid
            OSError: If directory cannot be created
        """
        if not path:
            raise ValueError("logs_dir path cannot be empty")
        
        logger.info(f"Changing logs directory to {path}")
        try:
            self._logs_dir = path
            self._logs_dir.mkdir(exist_ok=True)
            self._session_file = self._logs_dir / f"session_{self._session_id}.json"
        except Exception as e:
            logger.error(f"Failed to set logs directory to {path}: {e}")
            raise
    
    @property
    def session_file(self) -> Path:
        """Get the session file path"""
        return self._session_file
    
    @session_file.setter
    def session_file(self, path: Path):
        """
        Set the session file path and load data if it exists.
        
        Args:
            path: New session file path
            
        Raises:
            ValueError: If path is invalid
            OSError: If file operations fail
        """
        if not path:
            raise ValueError("session_file path cannot be empty")
        
        logger.info(f"Changing session file to {path}")
        old_file = self._session_file
        self._session_file = path
        
        try:
            # If we have data and the new file doesn't exist, save our data
            if old_file.exists() and not path.exists() and self._requests:
                logger.debug("Saving existing data to new session file")
                self._save_session()
            # If the new file exists, load its data
            elif path.exists():
                logger.debug("Loading data from existing session file")
                with open(path, 'r') as f:
                    data = json.load(f)
                    self._requests = data.get('requests', [])
                logger.info(f"Loaded {len(self._requests)} requests from {path}")
        except Exception as e:
            logger.error(f"Failed to handle session file change to {path}: {e}")
            raise
    
    @staticmethod
    def calculate_openai_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """
        Calculate cost for OpenAI API usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
            
        Returns:
            float: Cost in USD
        """
        # Cost per 1K tokens (as of March 2024)
        costs = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-4o": {"prompt": 0.01, "completion": 0.03},
            "gpt-4o-ms": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            "o1": {"prompt": 0.01, "completion": 0.03}
        }
        
        model_costs = costs.get(model, costs["gpt-3.5-turbo"])  # Default to gpt-3.5-turbo costs
        cost = (prompt_tokens * model_costs["prompt"] + completion_tokens * model_costs["completion"]) / 1000
        
        logger.debug("Calculated OpenAI cost", extra={
            "context": {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost
            }
        })
        
        return cost
    
    @staticmethod
    def calculate_claude_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """
        Calculate cost for Claude API usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
            
        Returns:
            float: Cost in USD
        """
        # Cost per 1M tokens (as of March 2024)
        costs = {
            "claude-3-opus-20240229": {"prompt": 15.0, "completion": 75.0},
            "claude-3-sonnet-20240229": {"prompt": 3.0, "completion": 15.0},
            "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
            "claude-3-5-sonnet-20241022": {"prompt": 3.0, "completion": 15.0}
        }
        
        model_costs = costs.get(model, costs["claude-3-sonnet-20240229"])  # Default to sonnet costs
        cost = (prompt_tokens * model_costs["prompt"] + completion_tokens * model_costs["completion"]) / 1_000_000
        
        logger.debug("Calculated Claude cost", extra={
            "context": {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost
            }
        })
        
        return cost
    
    def track_request(self, response: APIResponse):
        """
        Track a new API request.
        
        Args:
            response: API response information
            
        Raises:
            ValueError: If response is invalid
        """
        # Validate response
        if not response:
            raise ValueError("response cannot be None")
        
        # Only track costs for OpenAI and Anthropic
        if response.provider.lower() not in ["openai", "anthropic"]:
            logger.debug(f"Skipping cost tracking for unsupported provider: {response.provider}")
            return
        
        logger.debug(f"Tracking request for {response.provider} model {response.model}")
        request_data = {
            "timestamp": time.time(),
            "provider": response.provider,
            "model": response.model,
            "token_usage": {
                "prompt_tokens": response.token_usage.prompt_tokens,
                "completion_tokens": response.token_usage.completion_tokens,
                "total_tokens": response.token_usage.total_tokens,
                "reasoning_tokens": response.token_usage.reasoning_tokens
            },
            "cost": response.cost,
            "thinking_time": response.thinking_time
        }
        self._requests.append(request_data)
        
        try:
            self._save_session()
            logger.info(f"Request tracked successfully. Total requests: {len(self._requests)}")
        except Exception as e:
            logger.error(f"Failed to save session after tracking request: {e}")
            raise
    
    def get_session_summary(self) -> Dict:
        """
        Get summary of token usage and costs for the current session.
        
        Returns:
            Dict containing session statistics
        """
        logger.debug("Generating session summary")
        
        total_prompt_tokens = sum(r["token_usage"]["prompt_tokens"] for r in self._requests)
        total_completion_tokens = sum(r["token_usage"]["completion_tokens"] for r in self._requests)
        total_tokens = sum(r["token_usage"]["total_tokens"] for r in self._requests)
        total_cost = sum(r["cost"] for r in self._requests)
        total_thinking_time = sum(r["thinking_time"] for r in self._requests)
        
        # Group by provider
        provider_stats = {}
        for r in self._requests:
            provider = r["provider"]
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0
                }
            provider_stats[provider]["requests"] += 1
            provider_stats[provider]["total_tokens"] += r["token_usage"]["total_tokens"]
            provider_stats[provider]["total_cost"] += r["cost"]
        
        summary = {
            "total_requests": len(self._requests),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_thinking_time": total_thinking_time,
            "provider_stats": provider_stats,
            "session_duration": time.time() - self._session_start
        }
        
        logger.debug(f"Session summary: {len(self._requests)} requests, {total_tokens} tokens, ${total_cost:.6f}")
        return summary

# Global token tracker instance
_token_tracker: Optional[TokenTracker] = None

def get_token_tracker(session_id: Optional[str] = None, logs_dir: Optional[Path] = None) -> TokenTracker:
    """
    Get or create a global token tracker instance.
    
    Args:
        session_id: Optional session identifier
        logs_dir: Optional directory for log files
        
    Returns:
        TokenTracker: Global token tracker instance
        
    Raises:
        ValueError: If input parameters are invalid
        OSError: If log directory cannot be created or accessed
    """
    global _token_tracker
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.debug(f"Getting token tracker (session_id={session_id}, logs_dir={logs_dir})")
    
    # If no tracker exists, create one
    if _token_tracker is None:
        logger.debug("Creating new token tracker")
        _token_tracker = TokenTracker(session_id or current_date, logs_dir=logs_dir)
        return _token_tracker
    
    # If no session_id provided, reuse current tracker
    if session_id is None:
        logger.debug("Reusing existing token tracker")
        if logs_dir is not None:
            _token_tracker.logs_dir = logs_dir
        return _token_tracker
    
    # If session_id matches current tracker, reuse it
    if session_id == _token_tracker._session_id:
        logger.debug("Reusing existing token tracker with matching session_id")
        if logs_dir is not None:
            _token_tracker.logs_dir = logs_dir
        return _token_tracker
    
    # Otherwise, create a new tracker
    logger.debug("Creating new token tracker with different session_id")
    _token_tracker = TokenTracker(session_id, logs_dir=logs_dir)
    return _token_tracker

def load_session(session_file: Path) -> Optional[Dict]:
    """
    Load session data from a file.
    
    Args:
        session_file: Path to session file
        
    Returns:
        Optional[Dict]: Session data or None if file doesn't exist
        
    Raises:
        FileError: If file exists but cannot be read
    """
    if not session_file.exists():
        return None
    
    try:
        with open(session_file, 'r') as f:
            data = json.load(f)
            logger.debug("Loaded session file", extra={
                "context": {
                    "path": str(session_file),
                    "session_id": data.get("session_id")
                }
            })
            return data
    except Exception as e:
        logger.error("Failed to load session file", extra={
            "context": {
                "path": str(session_file),
                "error": str(e)
            }
        })
        raise FileError("Failed to load session file", str(session_file), {
            "error": str(e)
        })

def main():
    parser = argparse.ArgumentParser(
        description='Track and analyze token usage across LLM requests',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--session', help='Session ID to analyze')
    parser.add_argument('--logs-dir', type=Path, help='Directory for token tracking logs')
    
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
        tracker = get_token_tracker(args.session, args.logs_dir)
        summary = tracker.get_session_summary()
        
        metadata = {
            "session_id": tracker._session_id,
            "session_file": str(tracker.session_file),
            "start_time": format_timestamp(tracker._session_start)
        }
        
        print(format_output(summary, args.format, "Token Usage Summary", metadata))
        
    except (ValidationError, FileError) as e:
        logger.error(str(e), extra={"context": e.context})
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