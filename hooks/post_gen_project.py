import os
import shutil
import platform

def setup_env_file():
    """Set up the .env file with API key if provided"""
    llm_provider = '{{ cookiecutter["llm_provider [Optional. Press Enter to use None]"] }}'
    
    # If provider != 'None', retrieve whatever was saved in pre_gen_project.py
    if llm_provider != 'None':
        if os.path.exists(".temp_api_key"):
            with open(".temp_api_key", "r") as f:
                llm_api_key = f.read().strip()
            os.remove(".temp_api_key")

            if llm_api_key:
                provider_env_vars = {
                    'OpenAI': 'OPENAI_API_KEY',
                    'Anthropic': 'ANTHROPIC_API_KEY',
                    'DeepSeek': 'DEEPSEEK_API_KEY',
                    'Google': 'GOOGLE_API_KEY',
                    'Azure OpenAI': 'AZURE_OPENAI_API_KEY',
                    'Siliconflow': 'SILICONFLOW_API_KEY'
                }
                env_var_name = provider_env_vars.get(llm_provider)
                if env_var_name:
                    # Update .env or create it if needed
                    if not os.path.exists('.env'):
                        with open('.env', 'w') as _:
                            pass
                    with open('.env', 'r') as f:
                        lines = f.readlines()
                    with open('.env', 'w') as f:
                        key_found = False
                        for line in lines:
                            if line.startswith(env_var_name + '='):
                                f.write(f'{env_var_name}={llm_api_key}\n')
                                key_found = True
                            else:
                                f.write(line)
                        if not key_found:
                            f.write(f'{env_var_name}={llm_api_key}\n')

def handle_ide_rules():
    """Handle IDE-specific rules files based on project type"""
    project_type = '{{ cookiecutter.project_type }}'
    llm_provider = '{{ cookiecutter["llm_provider [Optional. Press Enter to use None]"] }}'
    
    # For Cursor projects: only keep .cursorrules
    if project_type == 'cursor':
        if os.path.exists('.windsurfrules'):
            os.remove('.windsurfrules')
        if os.path.exists('scratchpad.md'):
            os.remove('scratchpad.md')
        if os.path.exists('.github/copilot-instructions.md'):
            os.remove('.github/copilot-instructions.md')
        
        # Update .cursorrules if needed
        if os.path.exists('.cursorrules') and llm_provider == 'None':
            with open('.cursorrules', 'r') as f:
                content = f.readlines()
            
            # Find the Screenshot Verification section and insert the notice before it
            for i, line in enumerate(content):
                if '## Screenshot Verification' in line:
                    content.insert(i, '[NOTE TO CURSOR: Since no API key is configured, please ignore both the Screenshot Verification and LLM sections below.]\n')
                    content.insert(i + 1, '[NOTE TO USER: If you have configured or plan to configure an API key in the future, simply delete these two notice lines to enable these features.]\n\n')
                    break
            
            with open('.cursorrules', 'w') as f:
                f.writelines(content)
    
    # For Windsurf projects: keep both .windsurfrules and scratchpad.md
    elif project_type == 'windsurf':
        if os.path.exists('.cursorrules'):
            os.remove('.cursorrules')
        if os.path.exists('.github/copilot-instructions.md'):
            os.remove('.github/copilot-instructions.md')
        
        # Update .windsurfrules if needed
        if os.path.exists('.windsurfrules') and llm_provider == 'None':
            with open('.windsurfrules', 'r') as f:
                content = f.readlines()
            
            # Find the Screenshot Verification section and insert the notice before it
            for i, line in enumerate(content):
                if '## Screenshot Verification' in line:
                    content.insert(i, '[NOTE TO CURSOR: Since no API key is configured, please ignore both the Screenshot Verification and LLM sections below.]\n')
                    content.insert(i + 1, '[NOTE TO USER: If you have configured or plan to configure an API key in the future, simply delete these two notice lines to enable these features.]\n\n')
                    break
            
            with open('.windsurfrules', 'w') as f:
                f.writelines(content)
    
    # For GitHub Copilot projects: keep .github/copilot-instructions.md
    elif project_type == 'github copilot':
        if os.path.exists('.cursorrules'):
            os.remove('.cursorrules')
        if os.path.exists('.windsurfrules'):
            os.remove('.windsurfrules')
        if os.path.exists('scratchpad.md'):
            os.remove('scratchpad.md')
        
        # Update .github/copilot-instructions.md if needed
        if os.path.exists('.github/copilot-instructions.md') and llm_provider == 'None':
            with open('.github/copilot-instructions.md', 'r') as f:
                content = f.readlines()
            
            # Find the Screenshot Verification section and insert the notice before it
            for i, line in enumerate(content):
                if '## Screenshot Verification' in line:
                    content.insert(i, '[NOTE TO CURSOR: Since no API key is configured, please ignore both the Screenshot Verification and LLM sections below.]\n')
                    content.insert(i + 1, '[NOTE TO USER: If you have configured or plan to configure an API key in the future, simply delete these two notice lines to enable these features.]\n\n')
                    break
            
            with open('.github/copilot-instructions.md', 'w') as f:
                f.writelines(content)

def main():
    """Main function to set up the project"""
    setup_env_file()
    handle_ide_rules()
    
    # Create virtual environment
    print("\nCreating virtual environment...")
    os.system('python3 -m venv venv')
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if platform.system() == 'Windows':
        os.system('venv\\Scripts\\pip install -r requirements.txt')
    else:
        os.system('venv/bin/pip3 install -r requirements.txt')
    
    print("\nSetup completed successfully!")
    print("To get started:")
    print("1. Activate your virtual environment:")
    if platform.system() == 'Windows':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Check the README.md file for more information")

if __name__ == '__main__':
    main()