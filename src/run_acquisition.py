import sys
import os
import argparse
import pandas as pd

# Ensure the project directory is in the Python search path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import fetch_labeled_list, download_light_curves

def main():
    parser = argparse.ArgumentParser(description="Phase 1: High-Volume Data Acquisition")
    parser.add_argument("--num_per_class", type=int, default=700, help="Target count per astronomical class (default: 700)")
    parser.add_argument("--force_expand", action="store_true", help="Force expansion of the labels catalog")
    args = parser.parse_args()

    print("=== Phase 1: High-Volume Data Acquisition Started ===")
    
    labels_path = os.path.join("data", "labels.csv")
    raw_data_dir = os.path.join("data", "raw")
    failures_log_path = os.path.join("outputs", "download_failures.csv")
    progress_json_path = os.path.join("outputs", "download_progress.json")
    
    # Check if labels.csv already exists and has the requested targets
    labels_df = None
    if os.path.exists(labels_path):
        try:
            current_df = pd.read_csv(labels_path)
            # If current labels catalog is smaller than the requested target scale
            if len(current_df) < (args.num_per_class * 2.8) or args.force_expand:
                print(f"Catalog has {len(current_df)} targets. Expanding catalog to ~{args.num_per_class * 3} targets...")
                labels_df = fetch_labeled_list(num_per_class=args.num_per_class, existing_labels_path=labels_path)
                labels_df.to_csv(labels_path, index=False)
                print(f"Expanded labels catalog saved to {labels_path} ({len(labels_df)} stars).\n")
            else:
                labels_df = current_df
                print(f"Found existing labels file: {labels_path} with {len(labels_df)} targets.")
        except Exception as e:
            print(f"Error checking existing data cache: {e}. Re-fetching from archive...")
            labels_df = None

    if labels_df is None:
        labels_df = fetch_labeled_list(num_per_class=args.num_per_class)
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        labels_df.to_csv(labels_path, index=False)
        print(f"Labels CSV saved successfully to {labels_path} ({len(labels_df)} stars).\n")

    # Download missing light curves
    download_light_curves(
        labels_df=labels_df,
        output_dir=raw_data_dir,
        failures_path=failures_log_path,
        progress_json_path=progress_json_path
    )
    
    print("\n=== Phase 1: Data Acquisition Completed ===")

if __name__ == "__main__":
    main()
