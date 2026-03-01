import os, sys, time, subprocess
from pathlib import Path

def run():
    root = Path(__file__).parent
    py = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    npm = "npm.cmd" if os.name == "nt" else "npm"

    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", ".venv"])

    # Ensure dependencies are installed (idempotent)
    subprocess.run([str(py), "-m", "pip", "install", "-r", "backend/requirements.txt", "--quiet"])

    if not (root / "frontend/node_modules").exists():
        subprocess.run([npm, "install"], cwd="frontend", shell=True)

    print(">>> Starting Translatica...")
    be = subprocess.Popen([str(py), "-m", "uvicorn", "app.main:app", "--port", "8000"], 
                          cwd="backend", env={**os.environ, "PYTHONPATH": "backend"})
    fe = subprocess.Popen([npm, "run", "dev"], cwd="frontend", shell=True)

    try:
        while be.poll() is None and fe.poll() is None: time.sleep(1)
    except KeyboardInterrupt: pass
    finally:
        if os.name == 'nt':
            for p in [be, fe]: subprocess.call(f"taskkill /F /T /PID {p.pid}", stdout=-1, stderr=-1)
        else:
            be.terminate(); fe.terminate()

if __name__ == "__main__":
    run()
