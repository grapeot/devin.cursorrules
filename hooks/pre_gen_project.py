import re
import sys

# Get user input variables
project_name = '{{ cookiecutter.project_name }}'
project_type = '{{ cookiecutter.project_type }}'
llm_provider = '{{ cookiecutter["llm_provider [Optional. Press Enter to use None]"] }}'
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
    print("\nYou've chosen to enable advanced AI features with " + llm_provider + ".")
    print("These features include:")
    print("- Multi-modal analysis (e.g., screenshot verification for frontend development)")
    print("- Intelligent code analysis and suggestions")
    print("- Advanced debugging assistance")
    print("\nNote: These are optional features. You can:")
    print("1. Enter your API key now")
    print("2. Skip now and add it later in the .env file")
    print("3. Change your mind and not use these features at all")
    print("\nYour project will work perfectly fine without these features.")
    
    api_key_input = input(f"\nEnter your {llm_provider} API key (press Enter to skip): ")
    
    with open(".temp_api_key", "w") as f:
        f.write(api_key_input.strip()) 