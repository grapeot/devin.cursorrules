# Transform your $20 Cursor into a Devin-like AI Assistant

This repository gives you everything needed to supercharge your Cursor/Windsurf IDE or GitHub Copilot with **advanced** agentic AI capabilities—similar to the $500/month Devin—but at a fraction of the cost. In under a minute, you'll gain:

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
   Copy the `tools` folder and the following config files into your project root folder: Windsurf users need both `.windsurfrules` and `scratchpad.md` files. Cursor users only need the `.cursorrules` file. Github Copilot users need the `.github/copilot-instructions.md` file.

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

For a detailed walkthrough of setting up and using devin.cursorrules with Cursor, check out our [step-by-step tutorial](step_by_step_tutorial.md). This guide covers everything from initial Cursor setup to configuring devin.cursorrules and using the enhanced capabilities.

1. Choose your setup method:
   - **Cookiecutter (Recommended)**: Follow the prompts after running the cookiecutter command
   - **Manual**: Copy the files you need from this repository

2. Configure your environment:
   - Set up your API keys (optional)

3. Start exploring advanced tasks—such as data gathering, building quick prototypes, or cross-referencing external resources—in a fully agentic manner.

## Want the Details?

Check out our [blog post](https://yage.ai/cursor-to-devin-en.html) on how we turned $20 into $500-level AI capabilities in just one hour. It explains the philosophy behind process planning, self-evolution, and fully automated workflows. You'll also find side-by-side comparisons of Devin, Cursor, and Windsurf, plus a step-by-step tutorial on setting this all up from scratch.

License: MIT
