import subprocess, os, sys, time
from threading import Thread

def stream(pipe, label):
    for line in iter(pipe.readline, b''):
        if line: print(f"[{label}] {line.decode('utf-8', 'ignore').strip()}")

def start():
    root = os.path.dirname(os.path.abspath(__file__))
    frontend, backend = os.path.join(root, "frontend"), os.path.join(root, "backend")
    npm = "npm.cmd" if os.name == "nt" else "npm"

    # Quick check
    if not os.path.exists(os.path.join(frontend, "node_modules")):
        print("Installing frontend dependencies...")
        subprocess.run([npm, "install"], cwd=frontend, shell=True)

    print("\nStarting Translatica...")
    
    # Run Backend
    env = os.environ.copy()
    env["PYTHONPATH"] = backend
    be = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"], 
                          cwd=backend, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Run Frontend
    fe = subprocess.Popen([npm, "run", "dev"], cwd=frontend, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    Thread(target=stream, args=(be.stdout, "BACKEND"), daemon=True).start()
    Thread(target=stream, args=(fe.stdout, "FRONTEND"), daemon=True).start()

    try:
        while True:
            time.sleep(1)
            if be.poll() is not None or fe.poll() is not None: break
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        if os.name == 'nt':
            for p in [be, fe]: subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            be.terminate(); fe.terminate()
        print("Done.")

if __name__ == "__main__":
    start()
