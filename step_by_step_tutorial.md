# Step-by-Step Tutorial for Cursor with devin.cursorrules

This tutorial is designed for users who have never used Cursor before. We'll start from the beginning, covering installation, configuration, and how to use @grapeot's [`devin.cursorrules`](https://github.com/grapeot/devin.cursorrules) repository to transform Cursor into a self-evolving AI agent with tool-calling capabilities. While this document is designed for beginners, experienced Cursor users may also find it helpful. Feel free to skip sections you're already familiar with.

## Installation and Initial Configuration

Downloading and installing Cursor is similar to any other app. You can find the download link at the official website: [https://www.cursor.com/](https://www.cursor.com/). After launching Cursor for the first time, it will prompt you to log in. For first-time users, you'll need to click the register button to create an account on the official website.

To fully utilize Cursor, you'll need a Cursor Pro Plan subscription which costs $20 per month. However, Cursor provides a free trial period for new users. You can decide whether to subscribe after trying it out.

![Cursor Pro Plan](images/image2.png)

## Basic Interface

Cursor is a code editor where we typically open a folder to work in. For example, you can create a new folder like `~/Downloads/tmp` on your computer and use the "Open Folders" option in Cursor to open this location.

The interface consists of three main parts:
- Left sidebar: Shows the contents of your current folder (empty if you just created it)
- Middle area: Code editing space (though we'll primarily use Cursor's Agentic AI features)
- Right sidebar: Chat area where we communicate with Cursor, give instructions, and receive responses. If you don't see this area, press Command+I to show it.

![Basic Interface](images/image10.png)

Since we'll mainly use Cursor's Agentic AI features, I recommend making the chat sidebar wider.

Like VS Code, many of Cursor's features are accessed through commands in the command palette. You can press F1 to bring up the command palette. For example, if you can't remember how to bring up the chat panel, you can simply type "chat" in the command palette. It will show you options, and you can click the appropriate one to bring up the chat again. Commands also show keyboard shortcuts on the right, which you can memorize for faster access in the future.

![Command Palette](images/image4.png)

![Command Options](images/image9.png)

## Important Initial Settings

Cursor now provides a unified AI experience with Agent mode as the default. This means you no longer need to worry about switching between different modes like Chat, Composer, or Agent - there's just one smart interface that adapts to your needs.

In the bottom left corner of the chat panel, you can specify which AI model you want to use. Currently, Cursor supports several AI models including Claude, GPT-4o, and o3-mini. We generally recommend using Claude as it performs best in various scenarios, but feel free to experiment with other models.

Your configuration should look like this (note Claude in the bottom left):

![Configuration Settings](images/image8.png)

## YOLO Mode Configuration

Before we start our first example, we need to make one more configuration change. In the top right corner of the Cursor interface, there's a gear icon. Clicking it will take you to Cursor's settings. On the left side of the settings screen, there are four tabs: General, Models, Features, and Beta. Click the third tab (Features) and scroll down to "Enable Yolo Mode".

![YOLO Mode Settings](images/image5.png)

Here, you can configure based on your preferences:
- If you want to review and manually confirm every command before AI executes it, leave this unchecked
- If you trust the AI not to harm your system and want it to execute commands automatically, you can check this option

Below this, the Yolo Prompt allows you to further customize when AI can automatically execute commands. For example, you might write something like: "Ask for confirmation when the command involves file deletion, e.g. rm, rmdir, rsync --delete, find -delete"

## First Example: Stock Price Visualization

Now that we have configured Cursor properly, let's try our first example to see Cursor's AI agent capabilities in action. In the Composer panel, we can type a simple request like "plot the stock price of Google and Amazon in 2024 and show them in one figure".

At this point, Cursor will use its Agent mode to analyze the task, understand the requirements, and decide to use Python to complete this task.

![First Example Request](images/image1.png)

After Cursor automatically handles all the code writing, environment setup, and script execution, you'll see an image file generated in your current folder. When you click on this image file in the left sidebar, you'll see the stock price curves you requested.

![Stock Price Plot](images/image3.png)

This simple example demonstrates how Cursor's AI agent can understand natural language requests, write appropriate code, handle dependencies, and execute the code to produce the desired output, all without requiring you to write any code manually.

## Setting Up devin.cursorrules

Up to this point, we've been using Cursor's built-in features. While this AI agent is already powerful, it has several significant limitations: it can't self-evolve, can't remember learned experiences/lessons, and can't call some common external tools. To add these capabilities to Cursor, we can use @grapeot's repository: [https://github.com/grapeot/devin.cursorrules](https://github.com/grapeot/devin.cursorrules).

Here are the steps to configure and use this repo:

1. If you haven't installed Python yet, go to [https://www.python.org/downloads/](https://www.python.org/downloads/) or use your preferred package manager to install and configure Python.

2. Install the Cookiecutter dependency to easily initialize our Cursor project. In your system's command line (or Cursor's command window), run:
```bash
pip3 install cookiecutter
```

3. Go to where you want to place this Cursor project and execute this command:
```bash
cookiecutter gh:grapeot/devin.cursorrules --checkout template
```

If you get a "command not found: cookiecutter" error, try this command instead:
```bash
python3 -m cookiecutter gh:grapeot/devin.cursorrules --checkout template
```

It will launch a configuration wizard. Here is an example of the output:

```
➜  Downloads python3 -m cookiecutter gh:grapeot/devin.cursorrules --checkout template                
/Users/grapeot/Library/Python/3.9/lib/python/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
You've downloaded /Users/grapeot/.cookiecutters/devin.cursorrules before. Is it okay to delete and re-download it? [y/n] (y):
  [1/3] project_name (my-project): my-cursor-project
  [2/3] Select project_type
        1 - cursor
        2 - windsurf
        Choose from [1/2] (1):
  [3/3] Select llm_provider [Optional. Press Enter to use None]
        1 - None
        2 - OpenAI
        3 - Anthropic
        4 - DeepSeek
        5 - Google
        6 - Azure OpenAI
        Choose from [1/2/3/4/5/6] (1):
Creating virtual environment...
Installing dependencies...
```

The configuration has three steps:

1. Enter the name of your new project. Whatever name you enter, it will create a new subfolder with that name in the current directory and perform the configuration there.

2. Choose your project type. Currently, we support Cursor and Windsurf editors. Since we're using Cursor, just press Enter to select the default value (1).

3. Select an LLM Provider. This is an entirely optional configuration. When first starting, you can just press Enter to select None. It's only needed for some advanced features. We can start with None and come back to change it later when we're more familiar and need to use some advanced features.

The script will then automatically create the folder and configure the Python environment.

Next, you can use `cursor my-cursor-project` in the command line to open your newly created project, and you're ready to go.

## Using the Enhanced Tools

Using this enhanced Cursor project is similar to using a regular Cursor project, but now we have access to additional tools to better complete our tasks. For example, we can say "search recent news on OpenAI" in the prompt.

![Enhanced Tools Example](images/image6.png)

In this newly configured workspace, you'll notice Cursor has gained some additional capabilities. For instance, it will first edit our `.cursorrules` file for planning, then call our system search tools, and finally browse more web pages to get the latest information. 

Now you are ready to use the enhanced Cursor project to complete your other tasks!

![Tool Usage Example](images/image7.png)