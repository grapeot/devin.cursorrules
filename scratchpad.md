# Lessons

## User Specified Lessons

- You have a uv python venv in ./.venv. Always use it when running python scripts. It's a uv venv, so use `uv pip install` to install packages. And you need to activate it first. When you see errors like `no such file or directory: .venv/bin/uv`, that means you didn't activate the venv.
- Include info useful for debugging in the program output.
- Read the file before you try to edit it.
- Due to Windsurf's limit, when you use `git` and `gh` and need to submit a multiline commit message, first write the message in a file, and then use `git commit -F <filename>` or similar command to commit. And then remove the file. Include "[Windsurf] " in the commit message and PR title.

## Windsurf learned

- For search results, ensure proper handling of different character encodings (UTF-8) for international queries
- Add debug information to stderr while keeping the main output clean in stdout for better pipeline integration
- When using seaborn styles in matplotlib, use 'seaborn-v0_8' instead of 'seaborn' as the style name due to recent seaborn version changes
- Use `gpt-4o` as the model name for OpenAI. It is the latest GPT model and has vision capabilities as well. `o3` is the most advanced and expensive model from OpenAI. Use it when you need to do reasoning, planning, or get blocked.
- Use `claude-3-5-sonnet-20241022` as the model name for Claude. It is the latest Claude model and has vision capabilities as well.
- When running Python scripts that import from other local modules, use `PYTHONPATH=.` to ensure Python can find the modules. For example: `PYTHONPATH=. python tools/plan_exec_llm.py` instead of just `python tools/plan_exec_llm.py`. This is especially important when using relative imports.

# Multi-Agent Scratchpad

## Background and Motivation

(Planner writes: User/business requirements, macro objectives, why this problem needs to be solved)
The executor has access to three tools: invoking 3rd party LLM, invoking web browser, invoking search engine.

## Key Challenges and Analysis

(Planner: Records of technical barriers, resource constraints, potential risks)

## Verifiable Success Criteria

(Planner: List measurable or verifiable goals to be achieved)

## High-level Task Breakdown

(Planner: List subtasks by phase, or break down into modules)

## Current Status / Progress Tracking

(Executor: Update completion status after each subtask. If needed, use bullet points or tables to show Done/In progress/Blocked status)

## Next Steps and Action Items

(Planner: Specific arrangements for the Executor)

## Executor's Feedback or Assistance Requests

(Executor: Write here when encountering blockers, questions, or need for more information during execution)