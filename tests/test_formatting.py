#!/usr/bin/env python3

import unittest
import json
from datetime import datetime
from tools.common.formatting import (
    format_cost,
    format_duration,
    format_file_size,
    format_timestamp,
    format_output
)

class TestFormatting(unittest.TestCase):
    def test_format_cost(self):
        """Test cost formatting"""
        # Test zero cost
        self.assertEqual(format_cost(0), "$0.000000")
        
        # Test small cost
        self.assertEqual(format_cost(0.0001), "$0.000100")
        
        # Test larger cost
        self.assertEqual(format_cost(123.456789), "$123.456789")
        
        # Test negative cost
        with self.assertRaises(ValueError):
            format_cost(-1.0)

    def test_format_duration(self):
        """Test duration formatting"""
        # Test seconds
        self.assertEqual(format_duration(0), "0.00s")
        self.assertEqual(format_duration(45.678), "45.68s")
        
        # Test minutes
        self.assertEqual(format_duration(65), "1.08m")
        self.assertEqual(format_duration(1800), "30.00m")  # 30 minutes
        
        # Test hours
        self.assertEqual(format_duration(3600), "1.00h")  # 1 hour
        self.assertEqual(format_duration(7200), "2.00h")  # 2 hours
        
        # Test negative duration
        with self.assertRaises(ValueError):
            format_duration(-1)

    def test_format_file_size(self):
        """Test file size formatting"""
        # Test bytes
        self.assertEqual(format_file_size(0), "0.0B")
        self.assertEqual(format_file_size(100), "100.0B")
        
        # Test kilobytes
        self.assertEqual(format_file_size(1024), "1.0KB")
        self.assertEqual(format_file_size(2048), "2.0KB")
        
        # Test megabytes
        self.assertEqual(format_file_size(1024 * 1024), "1.0MB")
        self.assertEqual(format_file_size(2 * 1024 * 1024), "2.0MB")
        
        # Test gigabytes
        self.assertEqual(format_file_size(1024 * 1024 * 1024), "1.0GB")
        
        # Test terabytes
        self.assertEqual(format_file_size(1024 * 1024 * 1024 * 1024), "1.0TB")

    def test_format_timestamp(self):
        """Test timestamp formatting"""
        # Test specific timestamp
        timestamp = 1609459200  # 2021-01-01 00:00:00
        expected = "2021-01-01 00:00:00"
        self.assertEqual(format_timestamp(timestamp), expected)
        
        # Test current timestamp
        current = datetime.now().timestamp()
        formatted = format_timestamp(current)
        self.assertIsInstance(formatted, str)
        self.assertEqual(len(formatted), 19)  # YYYY-MM-DD HH:MM:SS

    def test_format_output_text(self):
        """Test text output formatting"""
        # Test string data
        result = format_output("Test string", format_type='text')
        self.assertEqual(result, "Test string")
        
        # Test dict data
        data = {"key1": "value1", "key2": "value2"}
        result = format_output(data, format_type='text', title="Test Title")
        self.assertIn("Test Title", result)
        self.assertIn("key1: value1", result)
        self.assertIn("key2: value2", result)
        
        # Test nested dict
        data = {"section": {"key1": "value1", "key2": "value2"}}
        result = format_output(data, format_type='text')
        self.assertIn("section:", result)
        self.assertIn("key1: value1", result)
        
        # Test list of dicts
        data = [
            {"key1": "value1"},
            {"key2": "value2"}
        ]
        result = format_output(data, format_type='text')
        self.assertIn("Result 1:", result)
        self.assertIn("Result 2:", result)
        self.assertIn("key1: value1", result)
        self.assertIn("key2: value2", result)
        
        # Test with metadata
        metadata = {"meta1": "value1"}
        result = format_output("Test", format_type='text', metadata=metadata)
        self.assertIn("meta1: value1", result)

    def test_format_output_json(self):
        """Test JSON output formatting"""
        # Test string data
        result = format_output("Test string", format_type='json')
        data = json.loads(result)
        self.assertEqual(data["data"], "Test string")
        
        # Test with title and metadata
        result = format_output(
            "Test string",
            format_type='json',
            title="Test Title",
            metadata={"meta1": "value1"}
        )
        data = json.loads(result)
        self.assertEqual(data["title"], "Test Title")
        self.assertEqual(data["metadata"]["meta1"], "value1")
        
        # Test complex data
        test_data = {
            "key1": "value1",
            "nested": {
                "key2": "value2"
            }
        }
        result = format_output(test_data, format_type='json')
        data = json.loads(result)
        self.assertEqual(data["data"]["key1"], "value1")
        self.assertEqual(data["data"]["nested"]["key2"], "value2")

    def test_format_output_markdown(self):
        """Test markdown output formatting"""
        # Test string data
        result = format_output("Test string", format_type='markdown')
        self.assertEqual(result, "Test string")
        
        # Test with title
        result = format_output("Test string", format_type='markdown', title="Test Title")
        self.assertIn("# Test Title", result)
        
        # Test dict data
        data = {"key1": "value1", "key2": "value2"}
        result = format_output(data, format_type='markdown')
        self.assertIn("**key1**: value1", result)
        self.assertIn("**key2**: value2", result)
        
        # Test nested dict
        data = {"section": {"key1": "value1"}}
        result = format_output(data, format_type='markdown')
        self.assertIn("## section", result)
        self.assertIn("**key1**: value1", result)
        
        # Test list of dicts
        data = [
            {"key1": "value1"},
            {"key2": "value2"}
        ]
        result = format_output(data, format_type='markdown')
        self.assertIn("## Result 1", result)
        self.assertIn("## Result 2", result)
        
        # Test with metadata
        metadata = {"meta1": "value1"}
        result = format_output("Test", format_type='markdown', metadata=metadata)
        self.assertIn("*meta1: value1*", result)

if __name__ == '__main__':
    unittest.main() 