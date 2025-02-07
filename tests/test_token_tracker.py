#!/usr/bin/env python3

import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
from pathlib import Path
import time
from datetime import datetime
from tools.token_tracker import TokenTracker, TokenUsage, APIResponse, get_token_tracker, _token_tracker
from tools.common.errors import ValidationError
import shutil

class TestTokenTracker(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_session_id = f"test-{int(time.time())}"
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
        self.tracker = TokenTracker(self.test_session_id, logs_dir=self.test_logs_dir)
        self.tracker.session_file = self.test_logs_dir / f"session_{self.test_session_id}.json"

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_logs_dir)
        
        # Reset global token tracker
        global _token_tracker
        _token_tracker = None

    def test_token_usage_creation(self):
        """Test TokenUsage creation and validation"""
        # Test valid token usage
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.completion_tokens, 50)
        self.assertEqual(usage.total_tokens, 150)
        self.assertIsNone(usage.reasoning_tokens)

        # Test with reasoning tokens
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, reasoning_tokens=20)
        self.assertEqual(usage.reasoning_tokens, 20)

        # Test invalid token counts
        with self.assertRaises(ValidationError):
            TokenUsage(prompt_tokens=-1, completion_tokens=50, total_tokens=150)
        with self.assertRaises(ValidationError):
            TokenUsage(prompt_tokens=100, completion_tokens=-1, total_tokens=150)
        with self.assertRaises(ValidationError):
            TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=-1)
        with self.assertRaises(ValidationError):
            TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, reasoning_tokens=-1)

    def test_api_response_creation(self):
        """Test APIResponse creation and validation"""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        
        # Test valid response
        response = APIResponse(
            content="Test response",
            token_usage=usage,
            cost=0.123,
            thinking_time=1.5,
            provider="openai",
            model="o1"
        )
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.token_usage, usage)
        self.assertEqual(response.cost, 0.123)
        self.assertEqual(response.thinking_time, 1.5)
        self.assertEqual(response.provider, "openai")
        self.assertEqual(response.model, "o1")

        # Test invalid responses
        with self.assertRaises(ValidationError):
            APIResponse(content="", token_usage=usage, cost=0.123)
        with self.assertRaises(ValidationError):
            APIResponse(content="Test", token_usage=usage, cost=-1)
        with self.assertRaises(ValidationError):
            APIResponse(content="Test", token_usage=usage, cost=0.123, thinking_time=-1)
        with self.assertRaises(ValidationError):
            APIResponse(content="Test", token_usage=usage, cost=0.123, provider="")
        with self.assertRaises(ValidationError):
            APIResponse(content="Test", token_usage=usage, cost=0.123, model="")

    def test_openai_cost_calculation(self):
        """Test OpenAI cost calculation"""
        # Test o1 model costs
        cost = TokenTracker.calculate_openai_cost(1000, 500, "o1")
        self.assertAlmostEqual(cost, 0.025)  # (1000 * 0.01 + 500 * 0.03) / 1000
        
        # Test gpt-4 model costs
        cost = TokenTracker.calculate_openai_cost(1000, 500, "gpt-4")
        self.assertAlmostEqual(cost, 0.06)  # (1000 * 0.03 + 500 * 0.06) / 1000
        
        # Test gpt-3.5-turbo model costs
        cost = TokenTracker.calculate_openai_cost(1000, 500, "gpt-3.5-turbo")
        self.assertAlmostEqual(cost, 0.00125)  # (1000 * 0.0005 + 500 * 0.0015) / 1000

    def test_claude_cost_calculation(self):
        """Test Claude cost calculation"""
        # Test Claude 3 Opus costs
        cost = TokenTracker.calculate_claude_cost(1000, 500, "claude-3-opus-20240229")
        self.assertAlmostEqual(cost, 0.0525)  # (1000 * 15 + 500 * 75) / 1_000_000
        
        # Test Claude 3 Sonnet costs
        cost = TokenTracker.calculate_claude_cost(1000, 500, "claude-3-sonnet-20240229")
        self.assertAlmostEqual(cost, 0.0105)  # (1000 * 3 + 500 * 15) / 1_000_000
        
        # Test Claude 3 Haiku costs
        cost = TokenTracker.calculate_claude_cost(1000, 500, "claude-3-haiku-20240307")
        self.assertAlmostEqual(cost, 0.000875)  # (1000 * 0.25 + 500 * 1.25) / 1_000_000

    def test_per_day_session_management(self):
        """Test session management with per-day sessions"""
        # Create tracker without session ID (should use current date)
        tracker = TokenTracker()
        tracker.logs_dir = self.test_logs_dir
        
        # Track a request
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        response = APIResponse(
            content="Test response",
            token_usage=usage,
            cost=0.123,
            thinking_time=1.5,
            provider="openai",
            model="o1"
        )
        tracker.track_request(response)
        
        # Verify request was tracked
        self.assertEqual(len(tracker._requests), 1)
        self.assertEqual(tracker._requests[0]["provider"], "openai")
        self.assertEqual(tracker._requests[0]["model"], "o1")
        self.assertEqual(tracker._requests[0]["token_usage"]["total_tokens"], 150)

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
        self.assertEqual(len(new_tracker._requests), 1)
        self.assertEqual(new_tracker._requests[0]["provider"], "openai")
        self.assertEqual(new_tracker._requests[0]["model"], "o1")
        self.assertEqual(new_tracker._requests[0]["token_usage"]["total_tokens"], 150)

    def test_session_summary_calculation(self):
        """Test session summary calculation"""
        tracker = TokenTracker(self.test_session_id)
        tracker.logs_dir = self.test_logs_dir
        
        # Track multiple requests
        for i in range(3):
            usage = TokenUsage(
                prompt_tokens=100 * (i + 1),
                completion_tokens=50 * (i + 1),
                total_tokens=150 * (i + 1)
            )
            response = APIResponse(
                content=f"Test response {i}",
                token_usage=usage,
                cost=0.123 * (i + 1),
                thinking_time=1.5 * (i + 1),
                provider="openai",
                model="o1"
            )
            tracker.track_request(response)
        
        # Get summary
        summary = tracker.get_session_summary()
        
        # Verify summary calculations
        self.assertEqual(len(tracker._requests), 3)
        self.assertEqual(summary["total_prompt_tokens"], 600)  # 100 + 200 + 300
        self.assertEqual(summary["total_completion_tokens"], 300)  # 50 + 100 + 150
        self.assertEqual(summary["total_tokens"], 900)  # 150 + 300 + 450
        self.assertAlmostEqual(summary["total_cost"], 0.738, places=3)  # 0.123 + 0.246 + 0.369
        self.assertAlmostEqual(summary["total_thinking_time"], 9.0, places=1)  # 1.5 + 3.0 + 4.5

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
        self.assertEqual(tracker3._session_id, "test-global-2")

if __name__ == "__main__":
    unittest.main() 