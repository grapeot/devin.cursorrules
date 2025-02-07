#!/usr/bin/env python3

from typing import Dict, Any, Optional

class ToolError(Exception):
    """Base class for all tool errors with standardized context"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize error with message and context.
        
        Args:
            message: Error message
            context: Optional context dictionary with error details
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        
    def __str__(self) -> str:
        """Return formatted error message with context if available"""
        if not self.context:
            return self.message
        
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message} ({context_str})"

class ValidationError(ToolError):
    """Error for input validation failures"""
    pass

class ConfigError(ToolError):
    """Error for configuration and environment issues"""
    pass

class APIError(ToolError):
    """Error for external API communication issues"""
    def __init__(self, message: str, provider: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize API error.
        
        Args:
            message: Error message
            provider: API provider name
            context: Optional context dictionary
        """
        context = context or {}
        context["provider"] = provider
        super().__init__(message, context)
        self.provider = provider

class FileError(ToolError):
    """Error for file system operations"""
    def __init__(self, message: str, path: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize file error.
        
        Args:
            message: Error message
            path: File path that caused the error
            context: Optional context dictionary
        """
        context = context or {}
        context["path"] = str(path)
        super().__init__(message, context)
        self.path = path 