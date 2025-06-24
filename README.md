# EPCalculatorLocal
Final Thesis project by Andreu Gordillo Vázquez, Alba Soldevila González, and Arnau Ranchal Marín, with the help and supervision of doctors Alfonso Martínez and Josep Font-Segura. 

This version is designed to be ran locally in a development environment.

---

## Prerequisites

Make sure you have the following installed on your system:

- **Python 3.8 or higher**
- **pip3** (Python package manager)
- **g++** (C++ compiler)
- **make** (build tool)

### Installing Prerequisites

#### On macOS:
```sh
xcode-select --install  # Installs g++ and make
brew install python3    # If Python 3 is not already installed
```

#### On Ubuntu/Debian:
```sh
sudo apt-get update
sudo apt-get install python3 python3-pip build-essential
```

#### On Windows:
- Install Visual Studio Build Tools
- Install Python 3 from python.org
- (Recommended) Install WSL2 for better compatibility

---

## Quick Setup

1. **Run the setup script:**
   ```sh
   chmod +x setup_local.sh
   ./setup_local.sh
   ```
   This script will:
   - Check if all prerequisites are installed
   - Install Python dependencies
   - Build the C++ library

2. **Start the application:**
   ```sh
   python3 run_local.py
   ```

3. **Access the application:**
   Open your browser and go to: [http://localhost:8000](http://localhost:8000)

4. *(Optional)* For the FastAPI Swagger docs, visit [http://localhost:8000/docs](http://localhost:8000/docs)

5. When finished, stop the server with `Ctrl+C` in the terminal

---

## Manual Setup (Alternative)

If you prefer to set up manually:

1. **Install Python dependencies:**
   ```sh
   pip3 install -r requirements.txt
   ```
2. **Build the C++ library:**
   ```sh
   make clean
   make
   ```
3. **Run the application:**
   ```sh
   python3 run_local.py
   ```

---

## Useful Commands

- **Recompile the C++ library if you change the C++ code:**
  ```sh
  make clean && make
  ```
- **Install Python dependencies:**
  ```sh
  pip3 install -r requirements.txt
  ```
- **Run the app (alternative):**
  ```sh
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
  ```

---

## Environment Variables

The application uses the following environment variables:

- `API_KEY`: Your OpenRouter API key (required for chatbot functionality)

You can set this in your shell:
```sh
export API_KEY="your-api-key-here"
```

Or create a `.env` file in the project root:
```
API_KEY=your-api-key-here
```

---

## Troubleshooting

### C++ Library Build Issues
- Make sure you have g++ and make installed
- On macOS, run `xcode-select --install` if you get build errors
- On Linux, install build-essential: `sudo apt-get install build-essential`

### Python Dependencies Issues
- Make sure you're using Python 3.8 or higher
- Try upgrading pip: `pip3 install --upgrade pip`
- If you get permission errors, use: `pip3 install --user -r requirements.txt`

### Port Already in Use
If port 8000 is already in use, you can modify `run_local.py` to use a different port:
```python
uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
```

---

## Development Features

The local development server includes:
- **Auto-reload**: Changes to Python files will automatically restart the server
- **Debug logging**: More detailed error messages
- **Local access**: Only accessible from localhost for security

---

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running to stop it.
