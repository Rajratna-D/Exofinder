import sys
import os
import pandas as pd

# Ensure the project directory is in the Python search path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import fetch_labeled_list, download_light_curves

def main():
    print("=== Phase 1: Data Acquisition Started ===")
    
    labels_path = os.path.join("data", "labels.csv")
    raw_data_dir = os.path.join("data", "raw")
    failures_log_path = os.path.join("outputs", "download_failures.csv")
    
    # Check if labels.csv already exists and has data, and check if all files in it exist in data/raw
    if os.path.exists(labels_path):
        try:
            labels_df = pd.read_csv(labels_path)
            if not labels_df.empty and 'star_id' in labels_df.columns:
                print(f"Found existing labels file: {labels_path} with {len(labels_df)} targets.")
                
                # Check if all corresponding light curve CSV files exist
                all_exist = True
                for star_id in labels_df['star_id']:
                    filepath = os.path.join(raw_data_dir, f"TIC_{int(star_id)}.csv")
                    if not (os.path.exists(filepath) and os.path.getsize(filepath) > 100):
                        all_exist = False
                        break
                        
                if all_exist:
                    print("All light curves already exist in data/raw/. Skipping network requests and download.")
                    print("\n=== Phase 1: Data Acquisition Completed ===")
                    return
                else:
                    print("Some light curves are missing or empty. Proceeding to download missing files using existing labels...")
                    download_light_curves(
                        labels_df=labels_df,
                        output_dir=raw_data_dir,
                        failures_path=failures_log_path
                    )
                    print("\n=== Phase 1: Data Acquisition Completed ===")
                    return
        except Exception as e:
            print(f"Error checking existing data cache: {e}. Proceeding to re-fetch from archive...")
    
    # 1. Query the Exoplanet Archive for 135 targets per class (total 405 stars)
    labels_df = fetch_labeled_list(num_per_class=135)
    
    # Save the labels file to data/labels.csv
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    labels_df.to_csv(labels_path, index=False)
    print(f"Labels CSV saved successfully to {labels_path}\n")
    
    # 2. Download the light curve data for each target star from MAST
    download_light_curves(
        labels_df=labels_df,
        output_dir=raw_data_dir,
        failures_path=failures_log_path
    )
    
    print("\n=== Phase 1: Data Acquisition Completed ===")

if __name__ == "__main__":
    main()
