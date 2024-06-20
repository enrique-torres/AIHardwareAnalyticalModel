import os
from typing import Dict, List

def list_directories(directory: str) -> List[str]:
    directories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return directories

def select_directory(directory: str) -> str:
    directories = list_directories(directory)
    if not directories:
        raise FileNotFoundError("No directories found in the specified directory.")
    print("Available options:")
    for i, dir_name in enumerate(directories):
        print(f"{i + 1}. {dir_name}")
    choice = int(input("Select an option by number: ")) - 1
    if choice < 0 or choice >= len(directories):
        raise ValueError("Invalid selection.")
    return os.path.join(directory, directories[choice])

def list_hardware_files(directory: str) -> List[str]:
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    return files

def select_hardware_model(directory: str) -> str:
    files = list_hardware_files(directory)
    if not files:
        raise FileNotFoundError("No hardware description files found in the directory.")
    print("Available hardware models:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    choice = int(input("Select a hardware model by number: ")) - 1
    if choice < 0 or choice >= len(files):
        raise ValueError("Invalid selection.")
    return os.path.join(directory, files[choice])