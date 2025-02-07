#!/usr/bin/env python3

import unittest
from unittest.mock import patch, MagicMock
import argparse
from tools.common.cli import add_common_args, create_parser, get_log_config
from tools.common.logging_config import LogLevel, LogFormat

class TestCLI(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.parser = argparse.ArgumentParser(description='Test Parser')

    def test_add_common_args(self):
        """Test adding common arguments to parser"""
        add_common_args(self.parser)
        
        # Parse with default arguments
        args = self.parser.parse_args([])
        self.assertEqual(args.format, 'text')
        self.assertEqual(args.log_level, 'info')
        self.assertEqual(args.log_format, 'text')
        self.assertFalse(args.debug)
        self.assertFalse(args.quiet)
        
        # Test format option
        args = self.parser.parse_args(['--format', 'json'])
        self.assertEqual(args.format, 'json')
        
        # Test log level options
        args = self.parser.parse_args(['--log-level', 'debug'])
        self.assertEqual(args.log_level, 'debug')
        
        args = self.parser.parse_args(['--debug'])
        self.assertTrue(args.debug)
        
        args = self.parser.parse_args(['--quiet'])
        self.assertTrue(args.quiet)
        
        # Test log format option
        args = self.parser.parse_args(['--log-format', 'json'])
        self.assertEqual(args.log_format, 'json')

    def test_create_parser(self):
        """Test parser creation"""
        # Test with common arguments
        parser = create_parser('Test Description')
        args = parser.parse_args([])
        self.assertEqual(args.format, 'text')
        self.assertEqual(args.log_level, 'info')
        
        # Test without common arguments
        parser = create_parser('Test Description', add_common=False)
        with self.assertRaises(SystemExit):
            # This should fail because no common arguments are added
            args = parser.parse_args(['--format', 'json'])
        
        # Test with custom formatter class
        parser = create_parser(
            'Test Description',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        # Create a formatter instance with required arguments
        formatter = parser.formatter_class(prog=parser.prog)
        self.assertIsInstance(formatter, argparse.RawDescriptionHelpFormatter)

    def test_get_log_config(self):
        """Test log configuration extraction"""
        parser = create_parser('Test Description')
        
        # Test default config
        args = parser.parse_args([])
        config = get_log_config(args)
        self.assertEqual(config['level'], LogLevel.INFO)
        self.assertEqual(config['format_type'], LogFormat.TEXT)
        
        # Test debug flag
        args = parser.parse_args(['--debug'])
        config = get_log_config(args)
        self.assertEqual(config['level'], LogLevel.DEBUG)
        
        # Test quiet flag
        args = parser.parse_args(['--quiet'])
        config = get_log_config(args)
        self.assertEqual(config['level'], LogLevel.QUIET)
        
        # Test explicit log level
        args = parser.parse_args(['--log-level', 'warning'])
        config = get_log_config(args)
        self.assertEqual(config['level'], LogLevel.WARNING)
        
        # Test log format
        args = parser.parse_args(['--log-format', 'json'])
        config = get_log_config(args)
        self.assertEqual(config['format_type'], LogFormat.JSON)

if __name__ == '__main__':
    unittest.main() 