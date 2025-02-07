#!/usr/bin/env python3

import unittest
from unittest.mock import patch, MagicMock
import logging
import json
import os
import sys
import threading
from tools.common.logging_config import (
    LogFormat,
    LogLevel,
    StructuredFormatter,
    JSONFormatter,
    setup_logging
)

class TestLoggingConfig(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Store original logging configuration
        self.original_loggers = dict(logging.root.manager.loggerDict)
        self.original_handlers = list(logging.root.handlers)

    def tearDown(self):
        """Clean up test fixtures"""
        # Restore original logging configuration
        logging.root.handlers = self.original_handlers
        for logger_name in list(logging.root.manager.loggerDict.keys()):
            if logger_name not in self.original_loggers:
                del logging.root.manager.loggerDict[logger_name]

    def test_log_level_conversion(self):
        """Test log level string to enum conversion"""
        # Test valid levels
        self.assertEqual(LogLevel.from_string("debug"), LogLevel.DEBUG)
        self.assertEqual(LogLevel.from_string("info"), LogLevel.INFO)
        self.assertEqual(LogLevel.from_string("warning"), LogLevel.WARNING)
        self.assertEqual(LogLevel.from_string("error"), LogLevel.ERROR)
        self.assertEqual(LogLevel.from_string("quiet"), LogLevel.QUIET)
        
        # Test case insensitivity
        self.assertEqual(LogLevel.from_string("DEBUG"), LogLevel.DEBUG)
        self.assertEqual(LogLevel.from_string("Info"), LogLevel.INFO)
        
        # Test invalid level
        with self.assertRaises(ValueError) as cm:
            LogLevel.from_string("invalid")
        self.assertIn("Invalid log level", str(cm.exception))
        self.assertIn("valid levels are", str(cm.exception).lower())

    def test_log_level_to_logging_level(self):
        """Test conversion to standard logging levels"""
        self.assertEqual(LogLevel.DEBUG.to_logging_level(), logging.DEBUG)
        self.assertEqual(LogLevel.INFO.to_logging_level(), logging.INFO)
        self.assertEqual(LogLevel.WARNING.to_logging_level(), logging.WARNING)
        self.assertEqual(LogLevel.ERROR.to_logging_level(), logging.ERROR)
        self.assertTrue(LogLevel.QUIET.to_logging_level() > logging.ERROR)

    def test_structured_formatter(self):
        """Test structured text formatter"""
        formatter = StructuredFormatter()
        
        # Create a test record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.correlation_id = "test-id"
        record.context = {"key": "value"}
        
        # Format record
        output = formatter.format(record)
        
        # Verify required components
        self.assertIn("[INFO]", output)
        self.assertIn("[test_logger]", output)
        self.assertIn("[test-id]", output)
        self.assertIn(f"[PID:{os.getpid()}]", output)
        self.assertIn(f"[TID:{threading.get_ident()}]", output)
        self.assertIn("[test.py:123]", output)
        self.assertIn("key=value", output)
        self.assertIn("Test message", output)
        
        # Test exception formatting
        try:
            raise ValueError("Test error")
        except ValueError:
            record.exc_info = sys.exc_info()
            output = formatter.format(record)
            self.assertIn("ValueError: Test error", output)

    def test_json_formatter(self):
        """Test JSON formatter"""
        formatter = JSONFormatter()
        
        # Create a test record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.correlation_id = "test-id"
        record.context = {"key": "value"}
        
        # Format record
        output = formatter.format(record)
        data = json.loads(output)
        
        # Verify required fields
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "test_logger")
        self.assertEqual(data["correlation_id"], "test-id")
        self.assertEqual(data["process_id"], os.getpid())
        self.assertEqual(data["thread_id"], threading.get_ident())
        self.assertEqual(data["file"], "test.py")
        self.assertEqual(data["line"], 123)
        self.assertEqual(data["message"], "Test message")
        self.assertEqual(data["context"]["key"], "value")
        
        # Test exception formatting
        try:
            raise ValueError("Test error")
        except ValueError:
            record.exc_info = sys.exc_info()
            output = formatter.format(record)
            data = json.loads(output)
            self.assertEqual(data["exception"]["type"], "ValueError")
            self.assertEqual(data["exception"]["message"], "Test error")
            self.assertIn("Traceback", data["exception"]["traceback"])

    def test_setup_logging(self):
        """Test logging setup"""
        # Test with string level
        logger = setup_logging("test_logger", level="debug")
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
        
        # Test with enum level
        logger = setup_logging("test_logger", level=LogLevel.INFO)
        self.assertEqual(logger.level, logging.INFO)
        
        # Test different formats
        logger = setup_logging("test_logger", format_type=LogFormat.JSON)
        self.assertIsInstance(logger.handlers[0].formatter, JSONFormatter)
        
        logger = setup_logging("test_logger", format_type=LogFormat.STRUCTURED)
        self.assertIsInstance(logger.handlers[0].formatter, StructuredFormatter)
        
        logger = setup_logging("test_logger", format_type=LogFormat.TEXT)
        self.assertIsInstance(logger.handlers[0].formatter, logging.Formatter)
        
        # Test handler replacement
        logger = setup_logging("test_logger")
        original_handler_count = len(logger.handlers)
        logger = setup_logging("test_logger")  # Setup again
        self.assertEqual(len(logger.handlers), original_handler_count)

if __name__ == '__main__':
    unittest.main() 