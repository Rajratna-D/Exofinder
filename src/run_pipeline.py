import sys
import os
import subprocess
import argparse
import shutil
import glob

# Define SafeStream to prevent crashes on closed stdout/stderr (e.g. during background task transitions)
class SafeStream:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        try:
            if self.stream is not None:
                self.stream.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            if self.stream is not None:
                self.stream.flush()
        except Exception:
            pass
    def fileno(self):
        try:
            if self.stream is not None:
                return self.stream.fileno()
        except Exception:
            pass
        raise OSError("No fileno")
    
    @property
    def closed(self):
        return False

    def __getattr__(self, name):
        if self.stream is not None:
            return getattr(self.stream, name)
        raise AttributeError(f"SafeStream has no underlying stream for attribute '{name}'")

sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)

# Add parent directory to path to enable local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_script(script_name):
    print(f"\n==========================================")
    print(f"RUNNING: {script_name}")
    print(f"==========================================")
    
    # Use the same python interpreter running this script
    python_exe = sys.executable
    script_path = os.path.join("src", script_name)
    
    # Run the subprocess redirecting stdout and stderr to a pipe so the child never crashes on closed standard handles.
    process = subprocess.Popen(
        [python_exe, "-u", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Read the output in real-time and print it safely to the parent's stdout
    while True:
        line = process.stdout.readline()
        if not line:
            break
        sys.stdout.write(line)
        sys.stdout.flush()
        
    process.wait()
    if process.returncode != 0:
        print(f"ERROR: {script_name} failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    else:
        print(f"SUCCESS: {script_name} completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="ExoFinder End-to-End Pipeline")
    parser.add_argument(
        "--custom_data_dir",
        type=str,
        default=None,
        help="Path to a directory containing custom light curve CSV files"
    )
    args = parser.parse_args()
    
    print("==========================================")
    print("EXOPLANET DETECTION END-TO-END PIPELINE")
    print("==========================================")
    
    skip_acquisition = False
    
    if args.custom_data_dir is not None:
        custom_dir = os.path.abspath(args.custom_data_dir)
        print(f"Loading custom dataset from: {custom_dir}\n")
        
        if not os.path.exists(custom_dir):
            print(f"ERROR: Custom data directory does not exist: {custom_dir}")
            sys.exit(1)
            
        # Copy CSV files to data/raw/
        os.makedirs("data/raw", exist_ok=True)
        csv_files = glob.glob(os.path.join(custom_dir, "*.csv"))
        
        # Check if custom directory contains a labels.csv file
        custom_labels = os.path.join(custom_dir, "labels.csv")
        if os.path.exists(custom_labels):
            shutil.copy(custom_labels, "data/labels.csv")
            print("Found labels.csv in custom directory, copied to data/labels.csv")
        else:
            # Auto-generate labels.csv from filename numeric IDs
            star_ids = []
            for f in csv_files:
                basename = os.path.basename(f)
                if basename == "labels.csv":
                    continue
                # Extract all digits
                digits = "".join(filter(str.isdigit, basename))
                if digits:
                    star_ids.append(int(digits))
                    
            if star_ids:
                import pandas as pd
                # Assign default valid class 'confirmed_planet' for classification label consistency
                df_labels = pd.DataFrame({
                    'star_id': star_ids,
                    'label': ['confirmed_planet'] * len(star_ids),
                    'catalog_period': [None] * len(star_ids)
                })
                df_labels.to_csv("data/labels.csv", index=False)
                print(f"Generated placeholder data/labels.csv for {len(star_ids)} stars.")
            else:
                print("WARNING: No valid light curves with numeric TIC IDs found in custom folder.")
                
        # Copy the actual light curve files
        copied_count = 0
        for f in csv_files:
            if os.path.basename(f) == "labels.csv":
                continue
            shutil.copy(f, os.path.join("data/raw", os.path.basename(f)))
            copied_count += 1
            
        print(f"Successfully copied {copied_count} light curves to data/raw/\n")
        skip_acquisition = True
        
    if not skip_acquisition:
        # Run standard acquisition stage
        run_script("run_acquisition.py")
        
    # Run remaining stages sequentially
    run_script("run_detrending.py")
    run_script("run_period_search.py")
    run_script("run_features.py")
    run_script("run_classifier.py")
    
    print("\n==========================================")
    print("END-TO-END PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("==========================================")
    print("To launch the interactive dashboard, run in your terminal:")
    print("  .venv\\Scripts\\streamlit run app.py")
    print("==========================================")

if __name__ == "__main__":
    main()
