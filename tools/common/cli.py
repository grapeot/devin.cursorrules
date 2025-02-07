#!/usr/bin/env python3

import argparse
from typing import Optional, List, Dict, Any
from .logging_config import LogLevel, LogFormat

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common arguments to an argument parser.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
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

def create_parser(
    description: str,
    *,
    add_common: bool = True,
    formatter_class: Optional[type] = None
) -> argparse.ArgumentParser:
    """
    Create an argument parser with optional common arguments.
    
    Args:
        description: Parser description
        add_common: Whether to add common arguments
        formatter_class: Optional formatter class
        
    Returns:
        argparse.ArgumentParser: Configured parser
    """
    if formatter_class is None:
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
        
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=formatter_class
    )
    
    if add_common:
        add_common_args(parser)
        
    return parser

def get_log_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Get logging configuration from parsed arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dict containing log_level and log_format
    """
    log_level = LogLevel.DEBUG if args.debug else (
        LogLevel.QUIET if args.quiet else 
        LogLevel.from_string(args.log_level)
    )
    log_format = LogFormat(args.log_format)
    
    return {
        "level": log_level,
        "format_type": log_format
    } 