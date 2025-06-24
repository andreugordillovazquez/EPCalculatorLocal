#!/bin/bash

echo "Setting up EPCalculator for local development..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip."
    exit 1
fi

# Check if g++ is installed
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ is not installed. Please install build tools."
    echo "On macOS: xcode-select --install"
    echo "On Ubuntu/Debian: sudo apt-get install build-essential"
    exit 1
fi

# Check if make is installed
if ! command -v make &> /dev/null; then
    echo "Error: make is not installed. Please install build tools."
    echo "On macOS: xcode-select --install"
    echo "On Ubuntu/Debian: sudo apt-get install build-essential"
    exit 1
fi

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Building C++ library..."
make clean
make

if [ $? -eq 0 ]; then
    echo "Setup completed successfully!"
    echo "You can now run the application with: python3 run_local.py"
else
    echo "Error: Failed to build C++ library. Please check the error messages above."
    exit 1
fi 