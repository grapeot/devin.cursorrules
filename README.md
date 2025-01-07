# Devin.cursorrules

Transform your $20 Cursor/Windsurf into a Devin-like experience in one minute! This repository contains configuration files and tools that enhance your Cursor or Windsurf IDE with advanced agentic AI capabilities similar to Devin, including:

- Process planning and self-evolution
- Extended tool usage (web browsing, search, LLM-powered analysis)
- Automated execution (for Windsurf in Docker containers)

## Usage

1. Copy all files from this repository to your project folder
2. For Cursor users: The `.cursorrules` file will be automatically loaded
3. For Windsurf users: Use both `.windsurfrules` and `scratchpad.md` for similar functionality

## Setup

1. Create Python virtual environment:
```bash
# Create a virtual environment in ./py310
python3 -m venv py310

# Activate the virtual environment
# On Unix/macOS:
source py310/bin/activate
# On Windows:
.\py310\Scripts\activate
```

2. Configure environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys and configurations
```

3. Install dependencies:
```bash
# Install required packages
pip install -r requirements.txt

# Install Playwright's Chromium browser (required for web scraping)
python -m playwright install chromium
```

## Tools Included

- Web scraping with JavaScript support (using Playwright)
- Search engine integration (DuckDuckGo)
- LLM-powered text analysis
- Process planning and self-reflection capabilities

## Testing

The project includes comprehensive unit tests for all tools. To run the tests:

```bash
# Make sure you're in the virtual environment
source py310/bin/activate  # On Windows: .\py310\Scripts\activate

# Run all tests
PYTHONPATH=. python -m unittest discover tests/
```

The test suite includes:
- Search engine tests (DuckDuckGo integration)
- Web scraper tests (Playwright-based scraping)
- LLM API tests (OpenAI integration)

## Background

For detailed information about the motivation and technical details behind this project, check out the blog post: [Turning $20 into $500 - Transforming Cursor into Devin in One Hour](https://yage.ai/cursor-to-devin-en.html)

## License

MIT License
