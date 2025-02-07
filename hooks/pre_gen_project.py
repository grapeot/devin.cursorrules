import re
import sys

# Get user input variables
project_name = '{{ cookiecutter.project_name }}'
project_type = '{{ cookiecutter.project_type }}'
llm_provider = '{{ cookiecutter.llm_provider }}'
valid_providers = ['None', 'OpenAI', 'Anthropic', 'DeepSeek', 'Google', 'Azure OpenAI']

# Validate project name
if not re.match(r'^[a-z][-a-z0-9]+$', project_name):
    print('ERROR: Project name must start with a letter and contain only lowercase letters, numbers, and hyphens.')
    sys.exit(1)

# Validate project type
if project_type not in ['cursor', 'windsurf']:
    print('ERROR: Project type must be either "cursor" or "windsurf".')
    sys.exit(1)

# Validate LLM provider
if llm_provider not in valid_providers:
    print(f'ERROR: LLM provider must be one of: {", ".join(valid_providers)}')
    sys.exit(1)

# Only ask for the key if provider is not 'None'
if llm_provider != 'None':
    print("\nNote: You can enter your LLM API key now or later by editing the .env file.")
    api_key_input = input(f"Enter your LLM API key for {llm_provider} (press Enter to skip): ")
    
    with open(".temp_api_key", "w") as f:
        f.write(api_key_input.strip()) 