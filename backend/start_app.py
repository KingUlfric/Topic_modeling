import subprocess
import webbrowser
import time
import os
import sys


def main():
    # Set the path to your project's backend directory
    project_path = "../backend"
    # Set the path to the Python executable in the virtual environment
    python_exec_path = 'C:/Users/DNS/AppData/Local/Programs/Python/Python310/python.exe'  # Use Scripts\python.exe on Windows

    # Change the current working directory to the project path
    os.chdir(project_path)

    # Set the PYTHONPATH environment variable to include the project directory
    sys.path.append(project_path)

    # Execute the FastAPI app using the Python executable from the virtual environment
    subprocess.call([python_exec_path, '-m', 'uvicorn', 'main:app', '--reload'])


def open_browser():
    """Open the web browser to the application URL."""
    webbrowser.open("http://127.0.0.1:8000/static/index.html")



if __name__ == "__main__":
    # Open the browser
    open_browser()
    # Add a slight delay to ensure the browser opens before the server starts
    time.sleep(5)
    # Start the server
    main()
