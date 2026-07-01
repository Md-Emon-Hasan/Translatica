import os, sys, time, json, socket, hashlib, shutil, threading, webbrowser, subprocess
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

# Everything is resolved relative to THIS file's folder (the repo root), never
# the shell's current directory -> `python run.py` works from anywhere.
ROOT = Path(__file__).resolve().parent

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"


def port_open(host, port):
    """True once something is accepting TCP connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def model_ready():
    """True once the backend reports the translation model has finished loading."""
    try:
        with urlopen(f"{BACKEND_URL}/health", timeout=2) as resp:
            return bool(json.loads(resp.read().decode()).get("model_loaded"))
    except (URLError, OSError, ValueError):
        return False


def wait_and_open(url, host, port, timeout=600):
    """Wait until the web server is up AND the model has finished loading, then
    open the browser -> the app is usable the moment the tab appears."""
    start = time.time()
    announced = False
    while time.time() - start < timeout:
        up = port_open(host, port)
        if up and model_ready():
            print(f">>> Ready! Model loaded. Opening {url} in your browser...")
            webbrowser.open(url)
            return
        if up and not announced:
            print(">>> Server is up; waiting for the translation model to finish "
                  "loading (this can take a bit on first run)...")
            announced = True
        time.sleep(0.5)
    print(">>> Timed out waiting for the model to load; open it manually:", url)


CORE_PACKAGES = ["uvicorn", "fastapi", "pydantic", "torch", "transformers",
                 "peft", "sqlalchemy", "sentencepiece"]


def deps_present(py):
    """Fast check (no import side effects) for whether core backend deps exist."""
    check = (
        "import importlib.util as u, sys;"
        f"mods={CORE_PACKAGES!r};"
        "missing=[m for m in mods if u.find_spec(m) is None];"
        "sys.exit(1 if missing else 0)"
    )
    return subprocess.run([str(py), "-c", check]).returncode == 0


def ensure_backend(py):
    """Create venv and install backend deps only when they are actually missing/changed."""
    req = ROOT / "backend" / "requirements.txt"
    stamp = ROOT / ".venv" / ".deps_stamp"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()

    # 1. Create the virtual environment on first run.
    if not py.exists():
        print(">>> Creating virtual environment (.venv)...")
        subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)

    # 2. Already verified this exact requirements.txt before -> nothing to do.
    if stamp.exists() and stamp.read_text().strip() == digest:
        return

    # 3. requirements.txt is new/changed (or first run). If the core packages are
    #    already importable, skip pip entirely and just record the stamp.
    if deps_present(py):
        stamp.write_text(digest)
        return

    print(">>> Installing backend dependencies (one-time, this can take a few minutes)...")
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    result = subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)])
    if result.returncode != 0:
        print(">>> ERROR: backend dependency install failed. Fix the error above and re-run.")
        sys.exit(1)
    stamp.write_text(digest)


def ensure_frontend(npm):
    """Install frontend deps if npm is available; otherwise skip gracefully."""
    if npm is None:
        return False
    if not (ROOT / "frontend" / "node_modules").exists():
        print(">>> Installing frontend dependencies (one-time, please wait)...")
        subprocess.run([npm, "install"], cwd=str(ROOT / "frontend"), shell=True)
    return True


def run():
    py = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")

    ensure_backend(py)
    have_frontend = ensure_frontend(npm)

    # On Windows, give children their own process group so Ctrl+C in this
    # console doesn't trigger the "Terminate batch job (Y/N)?" prompt from npm.
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

    print(">>> Starting Translatica...")
    print(f">>> Backend  -> {BACKEND_URL}")

    # Backend runs with the venv's python, from the backend/ dir (absolute path),
    # with backend/ on PYTHONPATH so `app.main` imports cleanly.
    backend_dir = str(ROOT / "backend")
    be = subprocess.Popen(
        [str(py), "-m", "uvicorn", "app.main:app", "--port", "8000"],
        cwd=backend_dir,
        env={**os.environ, "PYTHONPATH": backend_dir},
        creationflags=flags,
    )

    fe = None
    if have_frontend:
        print(f">>> Frontend -> {FRONTEND_URL}")
        fe = subprocess.Popen([npm, "run", "dev"], cwd=str(ROOT / "frontend"),
                              shell=True, creationflags=flags)
        # Open the UI once the frontend dev server is reachable.
        threading.Thread(target=wait_and_open,
                         args=(FRONTEND_URL, "localhost", 5173), daemon=True).start()
    else:
        print(">>> npm not found -> starting backend only.")
        print(f">>> Open the API docs at {BACKEND_URL}/docs")
        # No frontend: open the backend's Swagger UI instead.
        threading.Thread(target=wait_and_open,
                         args=(f"{BACKEND_URL}/docs", "localhost", 8000), daemon=True).start()

    print(">>> Running. Press Ctrl+C here to stop everything.")
    try:
        while be.poll() is None and (fe is None or fe.poll() is None):
            time.sleep(1)
        # Something exited on its own -> say which, so it isn't a mystery.
        if be.poll() is not None:
            print(f"\n>>> Backend stopped unexpectedly (exit code {be.returncode}).")
            print(">>> This is often 'out of memory' while loading torch.")
            print(">>> Fix: increase the Windows paging file, or close other apps "
                  "to free RAM, then re-run.")
        if fe is not None and fe.poll() is not None:
            print(f"\n>>> Frontend stopped unexpectedly (exit code {fe.returncode}).")
    except KeyboardInterrupt:
        pass
    finally:
        print("\n>>> Shutting down Translatica...")
        procs = [p for p in (be, fe) if p is not None]
        if os.name == 'nt':
            for p in procs:
                subprocess.call(f"taskkill /F /T /PID {p.pid}", stdout=-1, stderr=-1)
        else:
            for p in procs:
                p.terminate()


if __name__ == "__main__":
    run()
