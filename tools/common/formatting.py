#!/usr/bin/env python3

import json
from datetime import datetime
from typing import Dict, Any, List, Union
from pathlib import Path
from tabulate import tabulate

def format_cost(cost: float) -> str:
    """
    Format a cost value in dollars.
    
    Args:
        cost: Cost value in dollars
        
    Returns:
        str: Formatted cost string
        
    Raises:
        ValueError: If cost is negative
    """
    if cost < 0:
        raise ValueError("cost must be non-negative")
    return f"${cost:.6f}"

def format_duration(seconds: float) -> str:
    """
    Format duration in a human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
        
    Raises:
        ValueError: If seconds is negative
    """
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f}m"
    hours = minutes / 60
    return f"{hours:.2f}h"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"

def format_timestamp(timestamp: float) -> str:
    """
    Format Unix timestamp as human-readable date/time.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        str: Formatted timestamp string
    """
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def format_output(
    data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    format_type: str = 'text',
    title: str = None,
    metadata: Dict[str, Any] = None
) -> str:
    """
    Format data for output in a consistent way.
    
    Args:
        data: Data to format (string, dict, or list of dicts)
        format_type: Output format (text, json, or markdown)
        title: Optional title for the output
        metadata: Optional metadata to include
        
    Returns:
        str: Formatted output string
        
    Raises:
        ValidationError: If format_type is invalid
    """
    if format_type not in ['text', 'json', 'markdown']:
        raise ValidationError("Invalid output format", {
            "format": format_type,
            "valid_formats": ['text', 'json', 'markdown']
        })

    if format_type == 'json':
        output = {
            "data": data
        }
        if title:
            output["title"] = title
        if metadata:
            output["metadata"] = metadata
        return json.dumps(output, indent=2)
    
    elif format_type == 'markdown':
        output = []
        
        if title:
            output.extend([f"# {title}\n"])
        
        if isinstance(data, str):
            output.append(data)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    output.extend([f"## {key}", ""])
                    for k, v in value.items():
                        output.append(f"**{k}**: {v}")
                else:
                    output.append(f"**{key}**: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data, 1):
                output.extend([f"## Result {i}", ""])
                for key, value in item.items():
                    output.append(f"**{key}**: {value}")
                output.append("")
        
        if metadata:
            output.extend(["\n---"])
            for key, value in metadata.items():
                output.append(f"*{key}: {value}*")
                
        return "\n".join(output)
    
    else:  # text format
        output = []
        
        if title:
            output.extend([title, "=" * len(title), ""])
        
        if isinstance(data, str):
            output = [data]  # For string input, just return the string directly
            if metadata:  # Add metadata for string input too
                output.extend(["", "-" * 40])
                for key, value in metadata.items():
                    output.append(f"{key}: {value}")
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    output.extend([f"\n{key}:", ""])
                    for k, v in value.items():
                        output.append(f"{k}: {v}")
                else:
                    output.append(f"{key}: {value}")
            if metadata:
                output.extend(["", "-" * 40])
                for key, value in metadata.items():
                    output.append(f"{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data, 1):
                output.extend([f"\nResult {i}:", ""])
                for key, value in item.items():
                    output.append(f"{key}: {value}")
            if metadata:
                output.extend(["", "-" * 40])
                for key, value in metadata.items():
                    output.append(f"{key}: {value}")
                
        return "\n".join(output) 