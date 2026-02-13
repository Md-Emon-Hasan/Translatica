import subprocess
import sys
import signal
import os
import time
from threading import Thread

def stream_output(process, prefix):
    for line in iter(process.stdout.readline, b''):
        try:
            decoded_line = line.decode('utf-8').strip()
            print(f"[{prefix}] {decoded_line}")
        except UnicodeEncodeError:
            # Fallback for windows console that can't print utf-8 characters
            decoded_line = line.decode('utf-8', errors='ignore').strip()
            print(f"[{prefix}] {decoded_line.encode('ascii', 'ignore').decode('ascii')}")
        except Exception:
            pass

def run_services():
    # Paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")

    print("Starting Translatica Services...")

    # Start Backend (Uvicorn)
    # We need to set PYTHONPATH to include backend dir
    env = os.environ.copy()
    env["PYTHONPATH"] = backend_dir
    
    backend_cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    # If on windows, we might need shell=True for some commands, but list format is safer usually.
    # checking if uvicorn is in path or installed in environment. It should be.
    
    print(f"Starting Backend in {backend_dir}...")
    backend = subprocess.Popen(
        backend_cmd,
        cwd=backend_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Start Frontend (Vite)
    # npx vite or npm run dev
    # On windows npm is a cmd file
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    
    print(f"Starting Frontend in {frontend_dir}...")
    frontend = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True # Shell=True often needed for npm on windows to resolve paths correctly
    )

    # Threads to stream output
    Thread(target=stream_output, args=(backend, "BACKEND"), daemon=True).start()
    Thread(target=stream_output, args=(frontend, "FRONTEND"), daemon=True).start()

    try:
        while True:
            time.sleep(1)
            # Check if processes are alive
            if backend.poll() is not None:
                print("Backend process exited unexpectedly.")
                break
            if frontend.poll() is not None:
                print("Frontend process exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend.terminate()
        frontend.terminate()
        # On windows terminate might not be enough for shell=True processes
        if os.name == 'nt':
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(frontend.pid)])
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(backend.pid)])
        
        backend.wait()
        frontend.wait()
        print("Services stopped.")

if __name__ == "__main__":
    run_services()
