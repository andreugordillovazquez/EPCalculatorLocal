#!/usr/bin/env python3
"""
Local development server for EPCalculator
Run this script to start the application on localhost:8000
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Set development environment
    os.environ.setdefault("ENVIRONMENT", "development")
    
    # Check if the C++ library exists, if not, try to build it
    if not os.path.exists("./build/libfunctions.so"):
        print("C++ library not found. Attempting to build...")
        try:
            import subprocess
            result = subprocess.run(["make"], capture_output=True, text=True)
            if result.returncode != 0:
                print("Error building C++ library:")
                print(result.stderr)
                print("Please ensure you have g++ and make installed.")
                sys.exit(1)
            print("C++ library built successfully!")
        except FileNotFoundError:
            print("Error: 'make' command not found. Please install build tools.")
            print("On macOS: xcode-select --install")
            print("On Ubuntu/Debian: sudo apt-get install build-essential")
            sys.exit(1)
    
    # Start the server
    print("Starting EPCalculator on http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    ) 