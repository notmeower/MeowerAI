#!/usr/bin/env python3
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("1. train - Start training")
    print("2. api - Start API server")
    print("3. data - Load and process data")
    
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|api|data]")
        return
    
    command = sys.argv[1].lower()
    
    if command == "train":
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("\n" * 10)
        print(" " * 20 + "╔══════════════════════════════════════════════════════════════╗")
        print(" " * 20 + "║                                                              ║")
        print(" " * 20 + "║                    🚀 Starting Training 🚀                  ║")
        print(" " * 20 + "║                                                              ║")
        print(" " * 20 + "╚══════════════════════════════════════════════════════════════╝")
        print("\n" * 5)
        
        from main.train import main as train_main
        train_main()
    
    elif command == "api":
        print("Starting API server...")
        from api import run_server
        run_server()
    
    elif command == "data":
        print("Loading and processing data...")
        from main.data_loader import create_data_loader
        loader = create_data_loader()
        samples = loader.load_all_data()
        loader.save_processed_data("tensor/processed_data.json")
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, api, data")

if __name__ == "__main__":
    main()
