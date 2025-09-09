"""
Initialize test directories and files
"""

import os

def create_init_files():
    """Create __init__.py files for Python package structure"""
    
    directories = [
        "tests",
        "tests/model_compatibility", 
        "tests/model_compatibility/utils",
        "tests/model_compatibility/individual_model_tests",
        "tests/model_compatibility/results"
    ]
    
    for directory in directories:
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""Model compatibility testing framework"""')
            print(f"Created {init_file}")

if __name__ == "__main__":
    create_init_files()
