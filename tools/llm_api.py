#!/usr/bin/env /workspace/tmp_windsurf/py310/bin/python3

from openai import OpenAI
from anthropic import Anthropic
import argparse
import os
from dotenv import load_dotenv
from pathlib import Path

# 載入 .env.local 檔案
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
    elif provider == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return Anthropic(
            api_key=api_key
        )
    elif provider == "local":
        return OpenAI(
            base_url="http://192.168.180.137:8006/v1",
            api_key="not-needed"  # 本地部署可能不需要 API key
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def query_llm(prompt, client=None, model="", provider="openai"):
    if client is None:
        client = create_llm_client(provider)
    
    try:
        if provider == "openai":
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
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Query an LLM with a prompt')
    parser.add_argument('--prompt', type=str, help='The prompt to send to the LLM', required=True)
    parser.add_argument('--provider', type=str, choices=['openai', 'anthropic'], 
                       default="openai", help='The API provider to use')
    parser.add_argument('--model', type=str, 
                       help='The model to use (default depends on provider)')
    args = parser.parse_args()

    # 設定預設模型
    if not args.model:
        if args.provider == "openai":
            args.model = "gpt-3.5-turbo"
        else:
            args.model = "claude-3-5-sonnet-20241022"

    client = create_llm_client(args.provider)
    response = query_llm(args.prompt, client, model=args.model, provider=args.provider)
    if response:
        print(response)
    else:
        print("Failed to get response from LLM")

if __name__ == "__main__":
    main()
