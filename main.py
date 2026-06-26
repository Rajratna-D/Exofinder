import os
import sys
import subprocess

def main():
    print("==========================================")
    # Get current project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"ExoFinder Project Root: {project_dir}")
    print("==========================================")
    
    # Locate virtual environment candidates
    # 1. Check in project directory (exoplanet-detection-ps07/.venv)
    # 2. Check in parent directory (workspace root .venv)
    venv_dirs = [
        os.path.join(project_dir, ".venv"),
        os.path.join(os.path.dirname(project_dir), ".venv")
    ]
    
    python_exe = None
    for venv_dir in venv_dirs:
        # Windows venv path
        win_py = os.path.join(venv_dir, "Scripts", "python.exe")
        # Unix/macOS venv path
        unix_py = os.path.join(venv_dir, "bin", "python")
        
        if os.path.exists(win_py):
            python_exe = win_py
            break
        elif os.path.exists(unix_py):
            python_exe = unix_py
            break
            
    # Fallback to system python if no venv is found
    if python_exe is None:
        print("WARNING: Virtual environment .venv not found. Using default system python.")
        python_exe = sys.executable or "python"
    else:
        print(f"Using virtual environment Python: {python_exe}")
        
    # Step 1: Run the End-to-End Pipeline
    print("\n[Step 1/2] Running E2E Pipeline...")
    pipeline_script = os.path.join(project_dir, "src", "run_pipeline.py")
    ret = subprocess.run([python_exe, "-u", pipeline_script], cwd=project_dir)
    if ret.returncode != 0:
        print(f"\nERROR: Pipeline execution failed with code {ret.returncode}.")
        sys.exit(ret.returncode)
        
    # Step 2: Start Streamlit App
    print("\n[Step 2/2] Starting Streamlit Interactive Science Dashboard...")
    app_script = os.path.join(project_dir, "app.py")
    
    try:
        # Run streamlit app via 'python -m streamlit' which is extremely robust and does not rely on a separate wrapper file
        subprocess.run([python_exe, "-m", "streamlit", "run", app_script], cwd=project_dir)
    except KeyboardInterrupt:
        print("\nDashboard server stopped by user.")

if __name__ == "__main__":
    main()
