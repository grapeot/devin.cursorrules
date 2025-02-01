#!/usr/bin/env python3

import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
from pathlib import Path
import time
from datetime import datetime
from tools.token_tracker import TokenTracker, TokenUsage, APIResponse, get_token_tracker, _token_tracker

class TestTokenTracker(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test logs
        self.test_logs_dir = Path("test_token_logs")
        self.test_logs_dir.mkdir(exist_ok=True)
        
        # Clean up any existing test files
        for file in self.test_logs_dir.glob("*"):
            file.unlink()
        
        # Reset global token tracker
        global _token_tracker
        _token_tracker = None
        
        # Create test data
        self.test_token_usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=20
        )
        
        self.test_response = APIResponse(
            content="Test response",
            token_usage=self.test_token_usage,
            cost=0.123,
            thinking_time=1.5,
            provider="openai",
            model="o1"
        )
        
        # Create a TokenTracker instance with a unique test session ID
        self.test_session_id = f"test-{int(time.time())}"
        self.tracker = TokenTracker(self.test_session_id, logs_dir=self.test_logs_dir)
        self.tracker.session_file = self.test_logs_dir / f"session_{self.test_session_id}.json"

    def tearDown(self):
        # Clean up test logs directory
        if self.test_logs_dir.exists():
            for file in self.test_logs_dir.glob("*"):
                file.unlink()
            self.test_logs_dir.rmdir()
        
        # Reset global token tracker
        global _token_tracker
        _token_tracker = None

    def test_token_usage_creation(self):
        """Test TokenUsage dataclass creation"""
        token_usage = TokenUsage(100, 50, 150, 20)
        self.assertEqual(token_usage.prompt_tokens, 100)
        self.assertEqual(token_usage.completion_tokens, 50)
        self.assertEqual(token_usage.total_tokens, 150)
        self.assertEqual(token_usage.reasoning_tokens, 20)

    def test_api_response_creation(self):
        """Test APIResponse dataclass creation"""
        response = APIResponse(
            content="Test",
            token_usage=self.test_token_usage,
            cost=0.1,
            thinking_time=1.0,
            provider="openai",
            model="o1"
        )
        self.assertEqual(response.content, "Test")
        self.assertEqual(response.token_usage, self.test_token_usage)
        self.assertEqual(response.cost, 0.1)
        self.assertEqual(response.thinking_time, 1.0)
        self.assertEqual(response.provider, "openai")
        self.assertEqual(response.model, "o1")

    def test_openai_cost_calculation(self):
        """Test OpenAI cost calculation for supported models"""
        # Test o1 model pricing
        cost = TokenTracker.calculate_openai_cost(1000000, 500000, "o1")
        self.assertEqual(cost, 15.0 + 30.0)  # $15/M input + $60/M output
        
        # Test gpt-4o model pricing
        cost = TokenTracker.calculate_openai_cost(1000000, 500000, "gpt-4o")
        self.assertEqual(cost, 10.0 + 15.0)  # $10/M input + $30/M output
        
        # Test unsupported model
        with self.assertRaises(ValueError):
            TokenTracker.calculate_openai_cost(1000000, 500000, "gpt-4")

    def test_claude_cost_calculation(self):
        """Test Claude cost calculation"""
        cost = TokenTracker.calculate_claude_cost(1000000, 500000, "claude-3-sonnet-20240229")
        self.assertEqual(cost, 3.0 + 7.5)  # $3/M input + $15/M output

    def test_per_day_session_management(self):
        """Test per-day session management"""
        # Track a request
        self.tracker.track_request(self.test_response)
        
        # Verify file was created
        session_file = self.test_logs_dir / f"session_{self.test_session_id}.json"
        self.assertTrue(session_file.exists())
        
        # Load and verify file contents
        with open(session_file) as f:
            data = json.load(f)
            self.assertEqual(data["session_id"], self.test_session_id)
            self.assertEqual(len(data["requests"]), 1)
            self.assertEqual(data["requests"][0]["provider"], "openai")
            self.assertEqual(data["requests"][0]["model"], "o1")

    def test_session_file_loading(self):
        """Test loading existing session file"""
        # Create a test session file
        session_file = self.test_logs_dir / f"session_{self.test_session_id}.json"
        test_data = {
            "session_id": self.test_session_id,
            "start_time": time.time(),
            "requests": [
                {
                    "timestamp": time.time(),
                    "provider": "openai",
                    "model": "o1",
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                        "reasoning_tokens": 20
                    },
                    "cost": 0.123,
                    "thinking_time": 1.5
                }
            ]
        }
        with open(session_file, "w") as f:
            json.dump(test_data, f)
        
        # Create a new tracker - it should load the existing file
        new_tracker = TokenTracker(self.test_session_id)
        new_tracker.logs_dir = self.test_logs_dir
        new_tracker.session_file = self.test_logs_dir / f"session_{self.test_session_id}.json"
        self.assertEqual(len(new_tracker.requests), 1)
        self.assertEqual(new_tracker.requests[0]["provider"], "openai")
        self.assertEqual(new_tracker.requests[0]["model"], "o1")

    def test_session_summary_calculation(self):
        """Test session summary calculation"""
        # Add multiple requests with different providers
        responses = [
            APIResponse(
                content="Test 1",
                token_usage=TokenUsage(100, 50, 150, 20),
                cost=0.1,
                thinking_time=1.0,
                provider="openai",
                model="o1"
            ),
            APIResponse(
                content="Test 2",
                token_usage=TokenUsage(200, 100, 300, None),
                cost=0.2,
                thinking_time=2.0,
                provider="anthropic",
                model="claude-3-sonnet-20240229"
            )
        ]
        
        for response in responses:
            self.tracker.track_request(response)
        
        summary = self.tracker.get_session_summary()
        
        # Verify totals
        self.assertEqual(summary["total_requests"], 2)
        self.assertEqual(summary["total_prompt_tokens"], 300)
        self.assertEqual(summary["total_completion_tokens"], 150)
        self.assertEqual(summary["total_tokens"], 450)
        self.assertAlmostEqual(summary["total_cost"], 0.3, places=6)
        self.assertEqual(summary["total_thinking_time"], 3.0)
        
        # Verify provider stats
        self.assertEqual(len(summary["provider_stats"]), 2)
        self.assertEqual(summary["provider_stats"]["openai"]["requests"], 1)
        self.assertEqual(summary["provider_stats"]["anthropic"]["requests"], 1)

    def test_global_token_tracker(self):
        """Test global token tracker instance management"""
        # Get initial tracker with specific session ID
        tracker1 = get_token_tracker("test-global-1", logs_dir=self.test_logs_dir)
        self.assertIsNotNone(tracker1)
        
        # Get another tracker without session ID - should be the same instance
        tracker2 = get_token_tracker(logs_dir=self.test_logs_dir)
        self.assertIs(tracker1, tracker2)
        
        # Get tracker with different session ID - should be new instance
        tracker3 = get_token_tracker("test-global-2", logs_dir=self.test_logs_dir)
        self.assertIsNot(tracker1, tracker3)
        self.assertEqual(tracker3.session_id, "test-global-2")
        
        # Get tracker without session ID - should reuse the latest instance
        tracker4 = get_token_tracker(logs_dir=self.test_logs_dir)
        self.assertIs(tracker3, tracker4)

if __name__ == "__main__":
    unittest.main() 