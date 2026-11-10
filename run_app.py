"""Launch the Plant Disease app backend and open the correct frontend page."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
APP_MODULE = "backend.main:app"


def is_backend_running() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=2) as response:
            if response.status != 200:
                return False
        with urllib.request.urlopen(f"{URL}/admin", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError):
        return False


def stop_stale_process_on_port(port: int) -> bool:
    if os.name != "nt":
        return False

    try:
        netstat = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, check=False)
    except OSError:
        return False

    for line in netstat.stdout.splitlines():
        if f":{port}" not in line or "LISTENING" not in line.upper():
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        pid = parts[-1]
        if pid.isdigit() and pid != str(os.getpid()):
            subprocess.run(["taskkill", "/PID", pid, "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return True

    return False


def start_backend() -> subprocess.Popen:
    env = os.environ.copy()
    # Ensure the project root is on Python path so backend imports resolve correctly.
    env["PYTHONPATH"] = str(ROOT)

    command = [sys.executable, "-m", "uvicorn", APP_MODULE, "--host", HOST, "--port", str(PORT)]
    return subprocess.Popen(command, cwd=str(ROOT), env=env)


def wait_for_backend(timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    health_url = f"{URL}/health"

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError):
            time.sleep(0.5)
    return False


def main() -> int:
    if is_backend_running():
        print(f"Backend already responding at {URL}/health. Opening browser...")
        webbrowser.open(URL)
        return 0

    print("Starting LeafAI backend...")
    if stop_stale_process_on_port(PORT):
        print(f"Cleared a stale process occupying port {PORT}.")

    process = start_backend()

    try:
        print(f"Waiting for backend to become ready at {URL}/health...")
        if not wait_for_backend(20.0):
            print("ERROR: Backend did not start in time. Check Uvicorn output for errors.")
            return 1

        print(f"Backend is ready. Opening browser at {URL}")
        webbrowser.open(URL)
        print("Press CTRL+C to stop the backend.")

        process.wait()
        return process.returncode or 0

    except KeyboardInterrupt:
        print("\nShutting down backend...")
        return 0

    finally:
        if process.poll() is None:
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
