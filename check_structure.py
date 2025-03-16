import os
from pathlib import Path

required_files = [
    "hallucination_detection/__init__.py",
    "hallucination_detection/main.py",
    "hallucination_detection/llm.py",
    "hallucination_detection/debug_logger.py",
    "hallucination_detection/statement_parser.py",
    "hallucination_detection/check_aggregator.py",
    "hallucination_detection/checks/__init__.py",
    "hallucination_detection/samples/sample1.txt",
    "requirements.txt",
    "README.md",
    ".gitignore"
]

def check_structure():
    root = Path(__file__).parent
    missing = []
    for file in required_files:
        if not os.path.exists(os.path.join(root, file)):
            missing.append(file)
    return missing

if __name__ == "__main__":
    missing = check_structure()
    if missing:
        print("Missing required files:")
        for file in missing:
            print(f"  - {file}")
    else:
        print("All required files present!")
