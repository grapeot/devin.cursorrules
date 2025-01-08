#!/usr/bin/env /workspace/tmp_windsurf/py310/bin/python3

import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
import argparse
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('.') / '.env.local'
load_dotenv(dotenv_path=env_path)

def create_llm_client(provider="openai"):
    if provider == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(
            api_key=api_key
        )
    elif provider == "deepseek":
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
    elif provider == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return Anthropic(
            api_key=api_key
        )
    elif provider == "gemini":
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        return genai
    elif provider == "local":
        return OpenAI(
            base_url="http://192.168.180.137:8006/v1",
            api_key="not-needed"
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def query_llm(prompt, client=None, model=None, provider="openai"):
    if client is None:
        client = create_llm_client(provider)
    
    try:
        # Set default model
        if model is None:
            if provider == "openai":
                model = "gpt-3.5-turbo"
            elif provider == "deepseek":
                model = "deepseek-chat"
            elif provider == "anthropic":
                model = "claude-3-sonnet-20240229"
            elif provider == "gemini":
                model = "gemini-pro"
            elif provider == "local":
                model = "Qwen/Qwen2.5-32B-Instruct-AWQ"
            
        if provider == "openai" or provider == "local" or provider == "deepseek":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        elif provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        elif provider == "gemini":
            model = client.GenerativeModel(model)
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Query an LLM with a prompt')
    parser.add_argument('--prompt', type=str, help='The prompt to send to the LLM', required=True)
    parser.add_argument('--provider', choices=['openai','anthropic','gemini','local','deepseek'], default='openai', help='The API provider to use')
    parser.add_argument('--model', type=str, help='The model to use (default depends on provider)')
    args = parser.parse_args()

    if not args.model:
        if args.provider == 'openai':
            args.model = "gpt-3.5-turbo"
        elif args.provider == "deepseek":
            args.model = "deepseek-chat"
        elif args.provider == 'anthropic':
            args.model = "claude-3-5-sonnet-20241022"
        elif args.provider == 'gemini':
            args.model = "gemini-2.0-flash-exp"

    client = create_llm_client(args.provider)
    response = query_llm(args.prompt, client, model=args.model, provider=args.provider)
    if response:
        print(response)
    else:
        print("Failed to get response from LLM")

if __name__ == "__main__":
    main()
