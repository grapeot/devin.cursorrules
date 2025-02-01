# Transform your $20 Cursor into a Devin-like AI Assistant

This repository gives you everything needed to supercharge your Cursor or Windsurf IDE with **advanced** agentic AI capabilities — similar to the $500/month Devin—but at a fraction of the cost. In under a minute, you'll gain:

* Automated planning and self-evolution, so your AI "thinks before it acts" and learns from mistakes
* Extended tool usage, including web browsing, search engine queries, and LLM-driven text/image analysis
* [Experimental] Multi-agent collaboration, with o1 doing the planning, and regular Claude/GPT-4o doing the execution.

## Why This Matters

Devin impressed many by acting like an intern who writes its own plan, updates that plan as it progresses, and even evolves based on your feedback. But you don't need Devin's $500/month subscription to get most of that functionality. By customizing the .cursorrules file, plus a few Python scripts, you'll unlock the same advanced features inside Cursor.

## Key Highlights

1.	Easy Setup
   
   Copy the provided config files into your project folder. Cursor users only need the .cursorrules file. It takes about a minute, and you'll see the difference immediately.

2.	Planner-Executor Multi-Agent (Experimental)

   Our new [multi-agent branch](https://github.com/grapeot/devin.cursorrules/tree/multi-agent) introduces a high-level Planner (powered by o1) that coordinates complex tasks, and an Executor (powered by Claude/GPT) that implements step-by-step actions. This two-agent approach drastically improves solution quality, cross-checking, and iteration speed.

3.	Extended Toolset

   Includes:
   
   * Web scraping (Playwright)
   * Search engine integration (DuckDuckGo)
   * LLM-powered analysis

   The AI automatically decides how and when to use them (just like Devin).

4.	Self-Evolution

   Whenever you correct the AI, it can update its "lessons learned" in .cursorrules. Over time, it accumulates project-specific knowledge and gets smarter with each iteration. It makes AI a coachable and coach-worthy partner.
	
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
PYTHONPATH=. pytest -v tests/
```

Note: Use `-v` flag to see detailed test output including why tests were skipped (e.g. missing API keys)

The test suite includes:
- Search engine tests (DuckDuckGo integration)
- Web scraper tests (Playwright-based scraping)
- LLM API tests (OpenAI integration)

## Background

For detailed information about the motivation and technical details behind this project, check out the blog post: [Turning $20 into $500 - Transforming Cursor into Devin in One Hour](https://yage.ai/cursor-to-devin-en.html)

## License

MIT License
