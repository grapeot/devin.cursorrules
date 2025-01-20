# Devin.cursorrules

Transform your $20 Cursor/Windsurf into a Devin-like experience in one minute! This repository contains configuration files and tools that enhance your Cursor or Windsurf IDE with advanced agentic AI capabilities similar to Devin, including:

- Process planning and self-evolution
- Extended tool usage (web browsing, search, LLM-powered analysis)
- Automated execution (for Windsurf in Docker containers)

## Usage

1. Copy all files from this repository to your project folder
2. For Cursor users: The `.cursorrules` file will be automatically loaded
3. For Windsurf users: Use both `.windsurfrules` and `scratchpad.md` for similar functionality

## Update: Multi-Agent Support (Experimental)

This project includes experimental support for a multi-agent system that enhances Cursor's capabilities through a two-agent architecture:

### Architecture

- **Planner** (powered by OpenAI's o1 model): Handles high-level analysis, task breakdown, and strategic planning
- **Executor** (powered by Claude): Implements specific tasks, runs tests, and handles implementation details

[Actual .cursorrules file](https://github.com/grapeot/devin.cursorrules/blob/multi-agent/.cursorrules#L3)

### Key Benefits

1. **Enhanced Task Quality**
   - Separation of strategic planning from execution details
   - Better cross-checking and validation of solutions
   - Iterative refinement through Planner-Executor communication

2. **Improved Problem Solving**
   - Planner can design comprehensive test strategies
   - Executor provides detailed feedback and implementation insights
   - Continuous communication loop for optimization

### Real-World Example

A real case study of the multi-agent system debugging the DuckDuckGo search functionality:

1. **Initial Analysis**
   - Planner designed a series of experiments to investigate intermittent search failures
   - Executor implemented tests and collected detailed logs

2. **Iterative Investigation**
   - Planner analyzed results and guided investigation to the library's GitHub issues
   - Identified a bug in version 6.4 that was fixed in 7.2

3. **Solution Implementation**
   - Planner directed version upgrade and designed comprehensive test cases
   - Executor implemented changes and validated with diverse search scenarios
   - Final documentation included learnings and cross-checking measures

### Usage

To use the multi-agent system:

1. Switch to the `multi-agent` branch
2. The system will automatically coordinate between Planner and Executor roles
3. Planner uses `tools/plan_exec_llm.py` for high-level analysis
4. Executor implements tasks and provides feedback through the scratchpad

This experimental feature transforms the development experience from working with a single assistant to having both a strategic planner and a skilled implementer, significantly improving the depth and quality of task completion.

## Setup

1. Create Python virtual environment:
```bash
# Create a virtual environment in ./venv
python3 -m venv venv

# Activate the virtual environment
# On Unix/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
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
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

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
