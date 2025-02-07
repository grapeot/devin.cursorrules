# Transform your $20 Cursor into a Devin-like AI Assistant

This repository gives you everything needed to supercharge your Cursor or Windsurf IDE with **advanced** agentic AI capabilities—similar to the $500/month Devin—but at a fraction of the cost. In under a minute, you'll gain:

* Automated planning and self-evolution, so your AI "thinks before it acts" and learns from mistakes
* Extended tool usage, including web browsing, search engine queries, and LLM-driven text/image analysis
* [Experimental] Multi-agent collaboration, with o1 doing the planning, and regular Claude/GPT-4o doing the execution.

## Why This Matters

Devin impressed many by acting like an intern who writes its own plan, updates that plan as it progresses, and even evolves based on your feedback. But you don't need Devin's $500/month subscription to get most of that functionality. By customizing the .cursorrules file, plus a few Python scripts, you'll unlock the same advanced features inside Cursor.

## Key Highlights

1.	Easy Setup
   
   Two ways to get started:

   **Option 1: Using Cookiecutter (Recommended)**
   ```bash
   # Install cookiecutter if you haven't
   pip install cookiecutter

   # Create a new project
   cookiecutter gh:grapeot/devin.cursorrules --checkout template
   ```

   **Option 2: Manual Setup**
   Copy the provided config files into your project folder. Cursor users only need the .cursorrules file.

2.	Planner-Executor Multi-Agent (Experimental)

   Our new [multi-agent branch](https://github.com/grapeot/devin.cursorrules/tree/multi-agent) introduces a high-level Planner (powered by o1) that coordinates complex tasks, and an Executor (powered by Claude/GPT) that implements step-by-step actions. This two-agent approach drastically improves solution quality, cross-checking, and iteration speed.

3.	Extended Toolset

   Includes:
   
   * Web scraping (Playwright)
   * Search engine integration (DuckDuckGo)
   * LLM-powered analysis

   The AI automatically decides how and when to use them (just like Devin).

   Note: For screenshot verification features, Playwright browsers will be installed automatically when you first use the feature.

4.	Self-Evolution

   Whenever you correct the AI, it can update its "lessons learned" in .cursorrules. Over time, it accumulates project-specific knowledge and gets smarter with each iteration. It makes AI a coachable and coach-worthy partner.
	
## Usage

1. Choose your setup method:
   - **Cookiecutter (Recommended)**: Follow the prompts after running the cookiecutter command
   - **Manual**: Copy the files you need from this repository

2. Configure your environment:
   - Set up your API keys (optional)

3. Start exploring advanced tasks—such as data gathering, building quick prototypes, or cross-referencing external resources—in a fully agentic manner.

## Architecture

### .cursorrules File

The `.cursorrules` file is a crucial component of this architecture. It serves two main purposes:

1. **Maintaining Reusable Project Information**: This includes details like library versions, model names, and any corrections or fixes that have been made. By keeping this information in one place, the AI can avoid repeating past mistakes and ensure consistency across the project.

2. **Scratchpad for Task Planning**: The scratchpad section is used to organize thoughts and plan tasks. When a new task is received, the AI reviews the scratchpad, clears old tasks if necessary, explains the new task, and plans the steps needed to complete it. Progress is tracked using todo markers, and the scratchpad is updated as subtasks are completed.

### Development Container

The `.devcontainer/devcontainer.json` file sets up a development container with Python 3.10. This ensures a consistent development environment for all contributors.

### Environment Variables

Environment variables are managed through the `.env.example` file. This file includes placeholders for API keys and other configuration settings that are required for the project.

### Unit Tests

Unit tests are configured in the `.github/workflows/tests.yml` file. These tests run on pull requests and pushes to the main branches, ensuring that the codebase remains stable and functional.

### Tools

The project includes several tools that extend its capabilities:

1. **Screenshot Verification Workflow**: This workflow allows you to capture screenshots of web pages and verify their appearance using LLMs. The following tools are available:

   - Screenshot Capture:
     ```bash
     venv/bin/python tools/screenshot_utils.py URL [--output OUTPUT] [--width WIDTH] [--height HEIGHT]
     ```

   - LLM Verification with Images:
     ```bash
     venv/bin/python tools/llm_api.py --prompt "Your verification question" --provider {openai|anthropic} --image path/to/screenshot.png
     ```

   Example workflow:
   ```python
   from screenshot_utils import take_screenshot_sync
   from llm_api import query_llm

   # Take a screenshot
   screenshot_path = take_screenshot_sync('https://example.com', 'screenshot.png')

   # Verify with LLM
   response = query_llm(
       "What is the background color and title of this webpage?",
       provider="openai",  # or "anthropic"
       image_path=screenshot_path
   )
   print(response)
   ```

2. **LLM Integration**: The project includes functions for querying various LLM providers. For simple tasks, you can invoke the LLM by running the following command:
   ```bash
   venv/bin/python ./tools/llm_api.py --prompt "What is the capital of France?" --provider "anthropic"
   ```

   The LLM API supports multiple providers:
   - OpenAI (default, model: gpt-4o)
   - Azure OpenAI (model: configured via AZURE_OPENAI_MODEL_DEPLOYMENT in .env file, defaults to gpt-4o-ms)
   - DeepSeek (model: deepseek-chat)
   - Anthropic (model: claude-3-sonnet-20240229)
   - Gemini (model: gemini-pro)
   - Local LLM (model: Qwen/Qwen2.5-32B-Instruct-AWQ)

   Example usage:
   ```python
   from llm_api import query_llm

   response = query_llm("What is the capital of France?", provider="anthropic")
   print(response)
   ```

3. **Web Browser**: You can use the `tools/web_scraper.py` file to scrape the web.
   ```bash
   venv/bin/python ./tools/web_scraper.py --max-concurrent 3 URL1 URL2 URL3
   ```
   This will output the content of the web pages.

   Example usage:
   ```python
   from web_scraper import process_urls

   urls = ["https://example.com", "https://example.org"]
   results = process_urls(urls)
   for result in results:
       print(result)
   ```

4. **Search Engine**: You can use the `tools/search_engine.py` file to search the web.
   ```bash
   venv/bin/python ./tools/search_engine.py "your search keywords"
   ```
   This will output the search results in the following format:
   ```
   URL: https://example.com
   Title: This is the title of the search result
   Snippet: This is a snippet of the search result
   ```

   Example usage:
   ```python
   from search_engine import search

   results = search("your search keywords")
   for result in results:
       print(result)
   ```

### MCP Support from Claude's Model Context Protocol

The project also includes support for MCP (Model Context Protocol) from Claude's model context protocol. This allows for more advanced interactions and configurations with the LLMs, enhancing the overall capabilities of the project.

## Want the Details?

Check out our [blog post](https://yage.ai/cursor-to-devin-en.html) on how we turned $20 into $500-level AI capabilities in just one hour. It explains the philosophy behind process planning, self-evolution, and fully automated workflows. You'll also find side-by-side comparisons of Devin, Cursor, and Windsurf, plus a step-by-step tutorial on setting this all up from scratch.

License: MIT
