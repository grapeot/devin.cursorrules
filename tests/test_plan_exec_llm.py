#!/usr/bin/env python3

import unittest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import sys
import tempfile
import json
import pytest
import aiohttp
from tools.plan_exec_llm import (
    load_environment,
    read_plan_status,
    read_file_content,
    query_llm,
    read_content_or_file,
    validate_file_path,
    validate_plan,
    validate_execution_result,
    execute_plan,
    format_output
)
from tools.common.errors import ValidationError, FileError, APIError
from argparse import Namespace

# Add the parent directory to the Python path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.plan_exec_llm import TokenUsage

class AsyncContextManagerMock:
    def __init__(self, response):
        self.response = response
    
    async def __aenter__(self):
        return self.response
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class TestPlanExecLLM(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Save original environment
        self.original_env = dict(os.environ)
        # Set test environment variables
        os.environ['OPENAI_API_KEY'] = 'test_key'
        
        # Create a temporary test environment file
        self.env_file = '.env.test'
        with open(self.env_file, 'w') as f:
            f.write('OPENAI_API_KEY=test_key\n')
        
        # Create a temporary status file
        self.status_file = '.cursorrules'
        with open(self.status_file, 'w') as f:
            f.write('Some content\n# Multi-Agent Scratchpad\nTest content')
        
        # Patch the logger
        self.logger_patcher = patch('tools.plan_exec_llm.logger')
        self.mock_logger = self.logger_patcher.start()

    def tearDown(self):
        """Clean up test fixtures"""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up test files
        for file in [self.env_file, self.status_file, 'test_file.txt', 'test_dir']:
            try:
                if os.path.isdir(file):
                    os.rmdir(file)
                elif os.path.exists(file):
                    os.remove(file)
            except OSError as e:
                print(f"Warning: Failed to clean up {file}: {e}")
        
        # Ensure the status file is removed
        try:
            if os.path.exists(self.status_file):
                os.remove(self.status_file)
        except OSError as e:
            print(f"Warning: Failed to remove status file: {e}")
        
        self.logger_patcher.stop()

    def test_load_environment(self):
        """Test environment loading"""
        # Test with existing file
        env_loaded = load_environment()
        self.assertTrue(env_loaded)
        self.assertEqual(os.getenv('OPENAI_API_KEY'), 'test_key')

    def test_validate_file_path(self):
        """Test file path validation"""
        # Test with existing file
        path = validate_file_path(self.status_file)
        self.assertTrue(path.exists())
        
        # Test with non-existent file
        with self.assertRaises(FileError) as cm:
            validate_file_path('nonexistent_file.txt')
        self.assertIn("File not found", str(cm.exception))
        self.assertIn("nonexistent_file.txt", str(cm.exception))
        
        # Test with directory
        os.makedirs('test_dir', exist_ok=True)
        try:
            with self.assertRaises(FileError) as cm:
                validate_file_path('test_dir')
            self.assertIn("Not a file", str(cm.exception))
            self.assertIn("test_dir", str(cm.exception))
        finally:
            if os.path.exists('test_dir'):
                os.rmdir('test_dir')

    def test_read_plan_status(self):
        """Test reading plan status"""
        # Test with existing file
        content = read_plan_status()
        self.assertIn('Test content', content)
        
        # Test with missing section
        with open(self.status_file, 'w') as f:
            f.write("No scratchpad section")
        with self.assertRaises(ValidationError) as cm:
            read_plan_status()
        self.assertIn("section not found", str(cm.exception))
        
        # Test with empty section
        with open(self.status_file, 'w') as f:
            f.write("# Multi-Agent Scratchpad\n   ")
        with self.assertRaises(ValidationError) as cm:
            read_plan_status()
        self.assertIn("section is empty", str(cm.exception))
        
        # Test with missing file
        if os.path.exists(self.status_file):
            os.remove(self.status_file)
        with self.assertRaises(FileError) as cm:
            read_plan_status()
        self.assertIn("File not found", str(cm.exception))
        self.assertIn(".cursorrules", str(cm.exception))

    def test_read_file_content(self):
        """Test reading file content"""
        # Test with existing file
        content = read_file_content(self.env_file)
        self.assertIn('OPENAI_API_KEY=test_key', content)
    
        # Test with non-existent file
        with self.assertRaises(FileError) as cm:
            read_file_content('nonexistent_file.txt')
        self.assertIn("File not found", str(cm.exception))
        self.assertIn("nonexistent_file.txt", str(cm.exception))

    @patch('tools.plan_exec_llm.create_llm_client')
    def test_query_llm(self, mock_create_client):
        """Test LLM querying"""
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.completion_tokens_details = MagicMock()
        mock_response.usage.completion_tokens_details.reasoning_tokens = None
    
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_create_client.return_value = mock_client
    
        # Test with empty plan content
        with self.assertRaises(ValidationError) as cm:
            query_llm("")
        self.assertIn("Plan content cannot be empty", str(cm.exception))
        
        # Test with valid plan content
        response = query_llm("Test plan")
        self.assertEqual(response, "Test response")
        
        # Test error handling
        mock_client.chat.completions.create.side_effect = Exception("Test error")
        with self.assertRaises(APIError) as cm:
            query_llm("Test plan")
        self.assertIn("Failed to query LLM", str(cm.exception))

    def test_read_content_or_file(self):
        """Test reading content with @ prefix"""
        # Test direct string content
        content = read_content_or_file("Test content")
        self.assertEqual(content, "Test content")
        
        # Test empty content
        content = read_content_or_file("")
        self.assertEqual(content, "")
        
        # Test file content
        with open('test_file.txt', 'w') as f:
            f.write("File content")
        try:
            content = read_content_or_file("@test_file.txt")
            self.assertEqual(content, "File content")
        finally:
            if os.path.exists('test_file.txt'):
                os.remove('test_file.txt')
        
        # Test missing file
        with self.assertRaises(FileError) as cm:
            read_content_or_file("@nonexistent.txt")
        self.assertIn("Failed to read file", str(cm.exception))
        self.assertIn("nonexistent.txt", str(cm.exception))

    def test_format_output(self):
        """Test output formatting"""
        # Test text format
        result = format_output("Test output", "text")
        self.assertIn("Following is the instruction", result)
        self.assertIn("Test output", result)

        # Test JSON format
        result = format_output("Test output", "json")
        data = json.loads(result)
        self.assertEqual(data["response"], "Test output")
        self.assertEqual(data["model"], "o1")
        self.assertEqual(data["provider"], "openai")

        # Test markdown format
        result = format_output("Test output", "markdown")
        self.assertIn("# Plan Execution Response", result)
        self.assertIn("Test output", result)
        self.assertIn("*Model: o1 (OpenAI)*", result)

        # Test invalid format
        with self.assertRaises(ValidationError) as cm:
            format_output("Test output", "invalid")
        self.assertIn("Invalid format type", str(cm.exception))

    @pytest.mark.asyncio
    async def test_execute_plan(self):
        """Test plan execution"""
        # Test web search step
        plan = {
            "goal": "Test goal",
            "steps": [
                {
                    "id": "step1",
                    "name": "Search Step",
                    "type": "web_search",
                    "params": {
                        "query": "test query",
                        "max_results": 2
                    }
                }
            ]
        }
        
        # Mock search function
        with patch('tools.plan_exec_llm.search') as mock_search:
            mock_search.return_value = [
                {"url": "http://example.com", "title": "Test", "snippet": "Test snippet"}
            ]
            
            result = await execute_plan(plan)
            self.assertEqual(result["steps"][0]["status"], "success")
            self.assertIn("http://example.com", result["steps"][0]["output"])
        
        # Test web scrape step
        plan = {
            "goal": "Test goal",
            "steps": [
                {
                    "id": "step1",
                    "name": "Scrape Step",
                    "type": "web_scrape",
                    "params": {
                        "urls": ["http://example.com"]
                    }
                }
            ]
        }
        
        # Mock scraping function
        with patch('tools.plan_exec_llm.process_urls') as mock_scrape:
            mock_scrape.return_value = {
                "results": [{"url": "http://example.com", "content": "Test content"}],
                "errors": {}
            }
            
            result = await execute_plan(plan)
            self.assertEqual(result["steps"][0]["status"], "success")
            self.assertIn("Test content", result["steps"][0]["output"])
        
        # Test screenshot step
        plan = {
            "goal": "Test goal",
            "steps": [
                {
                    "id": "step1",
                    "name": "Screenshot Step",
                    "type": "screenshot",
                    "params": {
                        "url": "http://example.com"
                    }
                }
            ]
        }
        
        # Mock screenshot function
        with patch('tools.plan_exec_llm.take_screenshot') as mock_screenshot:
            mock_screenshot.return_value = "/tmp/test.png"
            
            result = await execute_plan(plan)
            self.assertEqual(result["steps"][0]["status"], "success")
            self.assertIn("/tmp/test.png", result["steps"][0]["output"])
        
        # Test step failure
        plan = {
            "goal": "Test goal",
            "steps": [
                {
                    "id": "step1",
                    "name": "Failed Step",
                    "type": "web_search",
                    "params": {
                        "query": "test query"
                    }
                }
            ]
        }
        
        with patch('tools.plan_exec_llm.search') as mock_search:
            mock_search.side_effect = APIError("Search failed", "search")
            
            result = await execute_plan(plan)
            self.assertEqual(result["steps"][0]["status"], "error")
            self.assertIn("Search failed", result["steps"][0]["error"])

    @patch('tools.plan_exec_llm.create_llm_client')
    def test_query_llm_with_system_prompt(self, mock_create_client):
        """Test LLM querying with system prompt"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.completion_tokens_details = MagicMock()
        mock_response.usage.completion_tokens_details.reasoning_tokens = 3
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_create_client.return_value = mock_client
        
        # Test with system prompt
        response = query_llm(
            "Test plan",
            system_prompt="You are a helpful assistant"
        )
        self.assertEqual(response, "Test response")
        
        # Verify system message was included
        calls = mock_client.chat.completions.create.call_args_list
        messages = calls[0][1]['messages']
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], 'You are a helpful assistant')

    @patch('argparse.ArgumentParser.parse_args')
    @patch('tools.plan_exec_llm.execute_plan')
    @patch('tools.plan_exec_llm.asyncio')
    @patch('tools.plan_exec_llm.validate_plan')
    @patch('tools.plan_exec_llm.read_content_or_file')
    async def test_main_function(self, mock_read, mock_validate, mock_execute, mock_parse_args):
        """Test main function execution"""
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.plan = '@test_plan.json'
        mock_args.format = 'text'
        mock_args.log_level = 'info'
        mock_args.log_format = 'text'
        mock_args.timeout = 300
        mock_parse_args.return_value = mock_args
        
        # Create test plan file
        plan = {
            "goal": "Test goal",
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "type": "web_search",
                    "params": {
                        "query": "test query"
                    }
                }
            ]
        }
        with open('test_plan.json', 'w') as f:
            json.dump(plan, f)
        
        # Mock plan execution
        mock_execute.return_value = {
            "total_steps": 1,
            "successful_steps": 1,
            "failed_steps": 0,
            "steps": [
                {
                    "id": "step1",
                    "status": "success",
                    "output": "Test output"
                }
            ]
        }
        mock_asyncio.run.return_value = mock_execute.return_value
        
        try:
            # Run main function
            with patch('sys.argv', ['plan_exec_llm.py', '@test_plan.json']):
                from tools.plan_exec_llm import main
                await main()
            
            # Verify plan was executed
            mock_asyncio.run.assert_called_once()
            
            # Verify calls
            mock_read.assert_called_once_with('test_plan.json')
            mock_validate.assert_called_once()
            mock_execute.assert_called_once()
            
        finally:
            # Clean up
            if os.path.exists('test_plan.json'):
                os.unlink('test_plan.json')

    def test_validate_execution_result(self):
        """Test execution result validation"""
        # Test valid result
        valid_result = {
            "step_id": "step1",
            "success": True,
            "output": "Test output"
        }
        validate_execution_result(valid_result)  # Should not raise
        
        # Test missing fields
        with self.assertRaises(ValidationError) as cm:
            validate_execution_result({})
        self.assertIn("Missing required result keys", str(cm.exception))
        
        # Test invalid status
        invalid_status = {
            "step_id": "step1",
            "success": "invalid",
            "output": "Test output"
        }
        with self.assertRaises(ValidationError) as cm:
            validate_execution_result(invalid_status)
        self.assertIn("Success must be a boolean", str(cm.exception))

    def test_validate_plan(self):
        """Test plan validation"""
        # Test valid plan
        valid_plan = {
            "goal": "Test goal",
            "steps": [
                {
                    "id": "step1",
                    "description": "Test step",
                    "action": "web_search",
                    "expected_result": "Search results",
                    "params": {
                        "query": "test query"
                    }
                }
            ]
        }
        validate_plan(valid_plan)  # Should not raise
        
        # Test missing plan keys
        with self.assertRaises(ValidationError) as cm:
            validate_plan({})
        self.assertIn("Missing required plan keys", str(cm.exception))
        
        # Test invalid steps type
        with self.assertRaises(ValidationError) as cm:
            validate_plan({"goal": "test", "steps": "invalid"})
        self.assertIn("Steps must be a list", str(cm.exception))
        
        # Test missing step keys
        invalid_step = {
            "goal": "test",
            "steps": [{"id": "step1"}]
        }
        with self.assertRaises(ValidationError) as cm:
            validate_plan(invalid_step)
        self.assertIn("Step 0 missing required keys", str(cm.exception))

if __name__ == '__main__':
    unittest.main() 