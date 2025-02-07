#!/usr/bin/env python3

import asyncio
from playwright.async_api import async_playwright, Error as PlaywrightError
import os
import tempfile
from pathlib import Path
import sys
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import time
from .common.logging_config import setup_logging
from .common.errors import ToolError, ValidationError, FileError
from .common.formatting import format_output, format_file_size, format_duration
from .common.cli import create_parser, get_log_config

logger = setup_logging(__name__)

class ScreenshotError(ToolError):
    """Custom exception for screenshot failures"""
    def __init__(self, message: str, url: str, details: Optional[Dict[str, Any]] = None):
        context = details or {}
        context["url"] = url
        super().__init__(message, context)
        self.url = url

def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False

def validate_dimensions(width: int, height: int) -> tuple[int, int]:
    """Validate viewport dimensions"""
    if width < 1 or height < 1:
        raise ValidationError("Width and height must be positive integers", {
            "width": width,
            "height": height
        })
    if width > 16384 or height > 16384:
        raise ValidationError("Width and height cannot exceed 16384 pixels", {
            "width": width,
            "height": height
        })
    return width, height

async def take_screenshot(url: str, output_path: str = None, width: int = 1280, height: int = 720) -> str:
    """
    Take a screenshot of a webpage using Playwright.
    
    Args:
        url: The URL to take a screenshot of
        output_path: Path to save the screenshot. If None, saves to a temporary file.
        width: Viewport width. Defaults to 1280.
        height: Viewport height. Defaults to 720.
    
    Returns:
        str: Path to the saved screenshot
        
    Raises:
        ValidationError: If input parameters are invalid
        ScreenshotError: If screenshot capture fails
        FileError: If output file cannot be created or written
    """
    # Validate inputs
    if not validate_url(url):
        logger.error("Invalid URL format", extra={
            "context": {
                "url": url,
                "error": "Invalid URL format"
            }
        })
        raise ValidationError("Invalid URL format", {"url": url})
    
    try:
        width, height = validate_dimensions(width, height)
    except ValidationError as e:
        logger.error("Invalid dimensions", extra={
            "context": e.context
        })
        raise
    
    # Prepare output path
    if output_path is None:
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            output_path = temp_file.name
            temp_file.close()
            logger.debug("Created temporary file", extra={
                "context": {
                    "path": output_path
                }
            })
        except OSError as e:
            logger.error("Failed to create temporary file", extra={
                "context": {
                    "error": str(e)
                }
            })
            raise FileError("Failed to create temporary file", "temp_file", {"error": str(e)})
    else:
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.debug("Created output directory", extra={
                    "context": {
                        "directory": output_dir
                    }
                })
            except OSError as e:
                logger.error("Failed to create output directory", extra={
                    "context": {
                        "directory": output_dir,
                        "error": str(e)
                    }
                })
                raise FileError("Failed to create output directory", output_dir, {"error": str(e)})
    
    logger.info("Taking screenshot", extra={
        "context": {
            "url": url,
            "output_path": output_path,
            "dimensions": f"{width}x{height}"
        }
    })
    start_time = time.time()
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page(viewport={'width': width, 'height': height})
                
                logger.debug("Navigating to page", extra={
                    "context": {
                        "url": url
                    }
                })
                await page.goto(url, wait_until='networkidle')
                
                logger.debug("Capturing screenshot", extra={
                    "context": {
                        "output_path": output_path
                    }
                })
                await page.screenshot(path=output_path, full_page=True)
                
                elapsed = time.time() - start_time
                file_size = os.path.getsize(output_path)
                logger.info("Screenshot captured successfully", extra={
                    "context": {
                        "url": url,
                        "output_path": output_path,
                        "elapsed_seconds": elapsed,
                        "file_size_bytes": file_size
                    }
                })
                return output_path
                
            except PlaywrightError as e:
                logger.error("Playwright error", extra={
                    "context": {
                        "url": url,
                        "error": str(e),
                        "elapsed_seconds": time.time() - start_time
                    }
                })
                raise ScreenshotError("Failed to capture screenshot", url, {
                    "error": str(e),
                    "elapsed_seconds": time.time() - start_time
                })
            finally:
                await browser.close()
                
    except Exception as e:
        # Clean up temporary file if we created one and failed
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass  # Ignore cleanup errors
        
        if isinstance(e, (ScreenshotError, ValidationError, FileError)):
            raise
        
        logger.error("Unexpected error during screenshot", extra={
            "context": {
                "url": url,
                "error": str(e),
                "elapsed_seconds": time.time() - start_time
            }
        })
        raise ScreenshotError("Screenshot capture failed", url, {
            "error": str(e),
            "elapsed_seconds": time.time() - start_time
        })

def take_screenshot_sync(url: str, output_path: str = None, width: int = 1280, height: int = 720) -> str:
    """
    Synchronous wrapper for take_screenshot.
    
    Args:
        url: The URL to take a screenshot of
        output_path: Path to save the screenshot. If None, saves to a temporary file.
        width: Viewport width. Defaults to 1280.
        height: Viewport height. Defaults to 720.
    
    Returns:
        str: Path to the saved screenshot
        
    Raises:
        ScreenshotError: If screenshot capture fails
        ValidationError: If input parameters are invalid
    """
    try:
        return asyncio.run(take_screenshot(url, output_path, width, height))
    except Exception as e:
        # Re-raise with appropriate type while preserving the traceback
        if isinstance(e, (ScreenshotError, ValidationError, FileError)):
            raise
        raise ScreenshotError("Screenshot capture failed", url, {
            "error": str(e)
        })

def main():
    parser = create_parser('Take a screenshot of a webpage')
    parser.add_argument('url', help='URL to take screenshot of')
    parser.add_argument('--output', '-o', help='Output path for screenshot')
    parser.add_argument('--width', '-w', type=int, default=1280, help='Viewport width')
    parser.add_argument('--height', '-H', type=int, default=720, help='Viewport height')
    
    args = parser.parse_args()

    # Configure logging
    log_config = get_log_config(args)
    logger = setup_logging(__name__, **log_config)
    logger.debug("Debug logging enabled", extra={
        "context": {
            "log_level": log_config["level"].value,
            "log_format": log_config["format_type"].value
        }
    })
    
    try:
        path = take_screenshot_sync(args.url, args.output, args.width, args.height)
        file_size = os.path.getsize(path)
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        
        result = {
            "screenshot_path": str(path),
            "file_size": format_file_size(file_size),
            "elapsed_time": format_duration(elapsed),
            "status": "success"
        }
        
        metadata = {
            "url": args.url,
            "dimensions": f"{args.width}x{args.height}"
        }
        
        print(format_output(result, args.format, "Screenshot Captured", metadata))
        
    except ValidationError as e:
        logger.error("Invalid input", extra={"context": e.context})
        sys.exit(1)
    except (ScreenshotError, FileError) as e:
        logger.error(str(e), extra={"context": e.context})
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", extra={
            "context": {
                "error": str(e)
            }
        })
        sys.exit(1)

if __name__ == "__main__":
    main() 