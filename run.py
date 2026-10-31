#!/usr/bin/env python3
"""
ExoFinder Nova — Unified Master Platform Launcher
=================================================
Launches both the FastAPI high-speed backend and the Vite React frontend
in synchronized subprocesses, monitors health, and opens the dashboard in your browser.

Usage:
    python run.py
"""

import os
import sys
import time
import signal
import socket
import webbrowser
import subprocess
import urllib.request
import json
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8001
FRONTEND_HOST = "localhost"
FRONTEND_PORT = 5174

BANNER = r"""
  ______             ______ _           _             _   _                 
 |  ____|           |  ____(_)         | |           | \ | |                
 | |__  __  _____   | |__   _ _ __   __| | ___ _ __  |  \| | _____   ____ _ 
 |  __| \ \/ / _ \  |  __| | | '_ \ / _` |/ _ \ '__| | . ` |/ _ \ \ / / _` |
 | |____ >  < (_) | | |    | | | | | (_| |  __/ |    | |\  | (_) \ V / (_| |
 |______/_/\_\___/  |_|    |_|_| |_|\__,_|\___|_|    |_| \_|\___/ \_/ \__,_|
                                                                            
             Deep Space Photometry & Keplerian Transit Intelligence
"""

def print_banner():
    print("\033[96m" + BANNER + "\033[0m")
    print(f"  \033[90mRoot Directory:\033[0m    {ROOT_DIR}")
    print(f"  \033[90mPython Binary:\033[0m     {VENV_PYTHON}")
    print(f"  \033[90mBackend Engine:\033[0m    http://{BACKEND_HOST}:{BACKEND_PORT}")
    print(f"  \033[90mFrontend UI:\033[0m       http://{FRONTEND_HOST}:{FRONTEND_PORT}\n")

def is_port_in_use(port: int, host: str = "127.0.0.1", timeout: float = 1.5) -> bool:
    for _ in range(2):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((host, port)) == 0:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False

def wait_for_service(url: str, timeout: float = 15.0, name: str = "Service") -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False

def main():
    print_banner()

    processes = []
    p_backend = None

    def shutdown(signum=None, frame=None):
        print("\n\033[93m[ExoFinder Nova] Shutting down platform services...\033[0m")
        for p in processes:
            try:
                if sys.platform == "win32":
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    p.terminate()
            except Exception:
                pass
        print("\033[92m[ExoFinder Nova] All processes stopped. Goodbye!\033[0m")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    # 1. Start FastAPI Backend if not already running
    backend_running = is_port_in_use(BACKEND_PORT, BACKEND_HOST)
    if not backend_running:
        print("\033[94m[1/3] Starting High-Speed FastAPI Backend Engine (port 8001)...\033[0m")
        backend_cmd = [
            str(VENV_PYTHON),
            "-m", "uvicorn",
            "src.server:app",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--log-level", "warning"
        ]
        p_backend = subprocess.Popen(backend_cmd, cwd=str(ROOT_DIR))
        processes.append(p_backend)

        if not wait_for_service(f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health", timeout=12.0, name="FastAPI"):
            print("\033[91m[Error] FastAPI backend failed to respond in time.\033[0m")
            shutdown()
        print("\033[92m  ✓ FastAPI Backend Online & Healthy\033[0m")
    else:
        print("\033[92m[1/3] FastAPI Backend is already running on port 8001.\033[0m")

    # 2. Start Vite React Frontend if not already running
    frontend_running = is_port_in_use(FRONTEND_PORT, "localhost") or is_port_in_use(5173, "localhost")
    active_frontend_port = FRONTEND_PORT
    if not frontend_running:
        print("\033[94m[2/3] Starting Vite React Sci-Fi Frontend (port 5174)...\033[0m")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        p_frontend = subprocess.Popen([npm_cmd, "run", "dev"], cwd=str(FRONTEND_DIR))
        processes.append(p_frontend)

        # Wait for either 5174 or 5173
        found = False
        for _ in range(30):
            if is_port_in_use(5174, "localhost"):
                active_frontend_port = 5174
                found = True
                break
            elif is_port_in_use(5173, "localhost"):
                active_frontend_port = 5173
                found = True
                break
            time.sleep(0.5)

        if found:
            print(f"\033[92m  ✓ Vite Frontend Online at http://localhost:{active_frontend_port}/\033[0m")
        else:
            print("\033[93m  ! Vite is starting up in background...\033[0m")
    else:
        active_frontend_port = 5174 if is_port_in_use(5174, "localhost") else 5173
        print(f"\033[92m[2/3] Vite Frontend is already active at http://localhost:{active_frontend_port}/\033[0m")

    # 3. Launch Default Web Browser
    dashboard_url = f"http://localhost:{active_frontend_port}/"
    print(f"\033[94m[3/3] Opening dashboard in your default browser: \033[96m{dashboard_url}\033[0m")
    time.sleep(0.8)
    try:
        webbrowser.open(dashboard_url)
    except Exception as e:
        print(f"  Note: Please open {dashboard_url} manually in your browser ({e}).")

    print("\n\033[97m========================================================================\033[0m")
    print(f"\033[92m  EXOFINDER NOVA IS LIVE & OPERATIONAL!\033[0m")
    print(f"  Dashboard UI:   \033[96m{dashboard_url}\033[0m")
    print(f"  API Docs (OpenAPI): \033[96mhttp://{BACKEND_HOST}:{BACKEND_PORT}/docs\033[0m")
    print(f"  Press \033[93mCtrl + C\033[0m in this terminal to terminate all services.")
    print("\033[97m========================================================================\033[0m\n")

    # Helper to safely monitor backend without spawning duplicate zombie processes
    def ensure_backend():
        nonlocal p_backend
        if p_backend is not None:
            # If backend process is still actively running, NEVER spawn a new one!
            if p_backend.poll() is None:
                return
            print(f"\033[93m[ExoFinder Nova] Backend process exited unexpectedly (code {p_backend.returncode}). Auto-recovering...\033[0m")
        else:
            # Backend wasn't started by this launcher; check if port 8001 is responding
            if is_port_in_use(BACKEND_PORT, BACKEND_HOST, timeout=2.0):
                return
            print("\033[93m[ExoFinder Nova] External backend disconnected. Auto-recovering FastAPI Engine (port 8001)...\033[0m")

        backend_cmd = [
            str(VENV_PYTHON),
            "-m", "uvicorn",
            "src.server:app",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--log-level", "warning"
        ]
        p_backend = subprocess.Popen(backend_cmd, cwd=str(ROOT_DIR))
        processes.append(p_backend)
        wait_for_service(f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health", timeout=8.0, name="FastAPI")
        print("\033[92m  ✓ FastAPI Backend Auto-Recovered & Online\033[0m")

    # Keep master runner alive with auto-healing
    try:
        check_counter = 0
        while True:
            time.sleep(1.0)
            check_counter += 1
            if check_counter % 5 == 0:
                ensure_backend()
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()
