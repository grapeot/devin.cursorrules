#!/usr/bin/env python3

import os
import time
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path
import uuid
import sys
from tabulate import tabulate
from datetime import datetime

@dataclass
class TokenUsage:
    """Token usage information for an LLM API request.
    
    Attributes:
        prompt_tokens: Number of tokens in the input prompt
        completion_tokens: Number of tokens in the model's response
        total_tokens: Total number of tokens used (prompt + completion)
        reasoning_tokens: Number of tokens used for reasoning (only available for OpenAI's o1 model)
            This is a special field that's only populated when using OpenAI's o1 model.
            For all other models (including other OpenAI models), this will be None.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None

@dataclass
class APIResponse:
    content: str
    token_usage: TokenUsage
    cost: float
    thinking_time: float = 0.0
    provider: str = "openai"
    model: str = "unknown"

class TokenTracker:
    def __init__(self, session_id: Optional[str] = None, logs_dir: Optional[Path] = None):
        # If no session_id provided, use today's date
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d")
        self.session_start = time.time()
        self.requests: List[Dict] = []
        
        # Create logs directory if it doesn't exist
        self._logs_dir = logs_dir or Path("token_logs")
        self._logs_dir.mkdir(exist_ok=True)
        
        # Initialize session file
        self._session_file = self._logs_dir / f"session_{self.session_id}.json"
        
        # Load existing session data if file exists
        if self._session_file.exists():
            try:
                with open(self._session_file, 'r') as f:
                    data = json.load(f)
                    self.session_start = data.get('start_time', self.session_start)
                    self.requests = data.get('requests', [])
            except Exception as e:
                print(f"Error loading existing session file: {e}", file=sys.stderr)
        
        self._save_session()
    
    def _save_session(self):
        """Save current session data to file"""
        session_data = {
            "session_id": self.session_id,
            "start_time": self.session_start,
            "requests": self.requests,
            "summary": self.get_session_summary()
        }
        with open(self._session_file, "w") as f:
            json.dump(session_data, f, indent=2)
    
    @property
    def logs_dir(self) -> Path:
        """Get the logs directory path"""
        return self._logs_dir
    
    @logs_dir.setter
    def logs_dir(self, path: Path):
        """Set the logs directory path and update session file path"""
        self._logs_dir = path
        self._logs_dir.mkdir(exist_ok=True)
        self.session_file = self._logs_dir / f"session_{self.session_id}.json"
    
    @property
    def session_file(self) -> Path:
        """Get the session file path"""
        return self._session_file
    
    @session_file.setter
    def session_file(self, path: Path):
        """Set the session file path and load data if it exists"""
        old_file = self._session_file
        self._session_file = path
        
        # If we have data and the new file doesn't exist, save our data
        if old_file.exists() and not path.exists() and self.requests:
            self._save_session()
        # If the new file exists, load its data
        elif path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.session_start = data.get('start_time', self.session_start)
                    self.requests = data.get('requests', [])
            except Exception as e:
                print(f"Error loading existing session file: {e}", file=sys.stderr)
    
    @staticmethod
    def calculate_openai_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate OpenAI API cost based on model and token usage"""
        # Only support o1, gpt-4o, and deepseek-chat models
        if model == "o1":
            # o1 pricing per 1M tokens
            INPUT_PRICE_PER_M = 15.0
            OUTPUT_PRICE_PER_M = 60.0
        elif model == "gpt-4o":
            # gpt-4o pricing per 1M tokens
            INPUT_PRICE_PER_M = 10.0
            OUTPUT_PRICE_PER_M = 30.0
        elif model == "deepseek-chat":
            # DeepSeek pricing per 1M tokens
            INPUT_PRICE_PER_M = 0.2  # $0.20 per million input tokens
            OUTPUT_PRICE_PER_M = 0.2  # $0.20 per million output tokens
        else:
            raise ValueError(f"Unsupported OpenAI model for cost calculation: {model}. Only o1, gpt-4o, and deepseek-chat are supported.")
        
        input_cost = (prompt_tokens / 1_000_000) * INPUT_PRICE_PER_M
        output_cost = (completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
        return input_cost + output_cost
    
    @staticmethod
    def calculate_claude_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate Claude API cost based on model and token usage"""
        # Claude-3 Sonnet pricing per 1M tokens
        # Source: https://www.anthropic.com/claude/sonnet
        if model in ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229"]:
            INPUT_PRICE_PER_M = 3.0  # $3 per million input tokens
            OUTPUT_PRICE_PER_M = 15.0  # $15 per million output tokens
        else:
            raise ValueError(f"Unsupported Claude model for cost calculation: {model}. Only claude-3-5-sonnet-20241022 and claude-3-sonnet-20240229 are supported.")
        
        input_cost = (prompt_tokens / 1_000_000) * INPUT_PRICE_PER_M
        output_cost = (completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
        return input_cost + output_cost
    
    def track_request(self, response: APIResponse):
        """Track a new API request"""
        # Only track costs for OpenAI and Anthropic
        if response.provider.lower() not in ["openai", "anthropic"]:
            return
            
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
        self.requests.append(request_data)
        self._save_session()
    
    def get_session_summary(self) -> Dict:
        """Get summary of token usage and costs for the current session"""
        total_prompt_tokens = sum(r["token_usage"]["prompt_tokens"] for r in self.requests)
        total_completion_tokens = sum(r["token_usage"]["completion_tokens"] for r in self.requests)
        total_tokens = sum(r["token_usage"]["total_tokens"] for r in self.requests)
        total_cost = sum(r["cost"] for r in self.requests)
        total_thinking_time = sum(r["thinking_time"] for r in self.requests)
        
        # Group by provider
        provider_stats = {}
        for r in self.requests:
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
        
        return {
            "total_requests": len(self.requests),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_thinking_time": total_thinking_time,
            "provider_stats": provider_stats,
            "session_duration": time.time() - self.session_start
        }

# Global token tracker instance
_token_tracker: Optional[TokenTracker] = None

def get_token_tracker(session_id: Optional[str] = None, logs_dir: Optional[Path] = None) -> TokenTracker:
    """Get or create a global token tracker instance"""
    global _token_tracker
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # If no tracker exists, create one
    if _token_tracker is None:
        _token_tracker = TokenTracker(session_id or current_date, logs_dir=logs_dir)
        return _token_tracker
    
    # If no session_id provided, reuse current tracker
    if session_id is None:
        if logs_dir is not None:
            _token_tracker.logs_dir = logs_dir
        return _token_tracker
    
    # If session_id matches current tracker, reuse it
    if session_id == _token_tracker.session_id:
        if logs_dir is not None:
            _token_tracker.logs_dir = logs_dir
        return _token_tracker
    
    # Otherwise, create a new tracker
    _token_tracker = TokenTracker(session_id, logs_dir=logs_dir)
    return _token_tracker

# Viewing functionality (moved from view_usage.py)
def format_cost(cost: float) -> str:
    """Format a cost value in dollars"""
    return f"${cost:.6f}"

def format_duration(seconds: float) -> str:
    """Format duration in a human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f}m"
    hours = minutes / 60
    return f"{hours:.2f}h"

def load_session(session_file: Path) -> Optional[Dict]:
    """Load a session file and return its contents"""
    try:
        with open(session_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading session file {session_file}: {e}", file=sys.stderr)
        return None

def display_session_summary(session_data: Dict, show_requests: bool = False):
    """Display a summary of the session"""
    summary = session_data["summary"]
    
    # Print session overview
    print("\nSession Overview")
    print("===============")
    print(f"Session ID: {session_data['session_id']}")
    print(f"Duration: {format_duration(summary['session_duration'])}")
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Total Cost: {format_cost(summary['total_cost'])}")
    
    # Print token usage
    print("\nToken Usage")
    print("===========")
    print(f"Prompt Tokens: {summary['total_prompt_tokens']:,}")
    print(f"Completion Tokens: {summary['total_completion_tokens']:,}")
    print(f"Total Tokens: {summary['total_tokens']:,}")
    
    # Print provider stats
    print("\nProvider Statistics")
    print("==================")
    provider_data = []
    for provider, stats in summary["provider_stats"].items():
        provider_data.append([
            provider,
            stats["requests"],
            f"{stats['total_tokens']:,}",
            format_cost(stats["total_cost"])
        ])
    print(tabulate(
        provider_data,
        headers=["Provider", "Requests", "Tokens", "Cost"],
        tablefmt="simple"
    ))
    
    # Print individual requests if requested
    if show_requests:
        print("\nIndividual Requests")
        print("==================")
        request_data = []
        for req in session_data["requests"]:
            request_data.append([
                req["provider"],
                req["model"],
                f"{req['token_usage']['total_tokens']:,}",
                format_cost(req["cost"]),
                f"{req['thinking_time']:.2f}s"
            ])
        print(tabulate(
            request_data,
            headers=["Provider", "Model", "Tokens", "Cost", "Time"],
            tablefmt="simple"
        ))

def list_sessions(logs_dir: Path):
    """List all available session files"""
    session_files = sorted(logs_dir.glob("session_*.json"))
    if not session_files:
        print("No session files found.")
        return
    
    for session_file in session_files:
        session_data = load_session(session_file)
        if session_data:
            summary = session_data["summary"]
            print(f"\nSession: {session_data['session_id']}")
            print(f"Duration: {format_duration(summary['session_duration'])}")
            print(f"Requests: {summary['total_requests']}")
            print(f"Total Cost: {format_cost(summary['total_cost'])}")
            print(f"Total Tokens: {summary['total_tokens']:,}")

def main():
    parser = argparse.ArgumentParser(description='View LLM API usage statistics')
    parser.add_argument('--session', type=str, help='Session ID to view details for')
    parser.add_argument('--requests', action='store_true', help='Show individual requests')
    args = parser.parse_args()
    
    logs_dir = Path("token_logs")
    if not logs_dir.exists():
        print("No logs directory found")
        return
    
    if args.session:
        session_file = logs_dir / f"session_{args.session}.json"
        if not session_file.exists():
            print(f"Session file not found: {session_file}")
            return
        
        session_data = load_session(session_file)
        if session_data:
            display_session_summary(session_data, args.requests)
    else:
        list_sessions(logs_dir)

if __name__ == "__main__":
    main() 