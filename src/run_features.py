import sys
import os

# Add parent directory to path to enable local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import extract_features_all

def main():
    print("=== Phase 5: Feature Extraction Started ===")
    extract_features_all()
    print("=== Phase 5: Feature Extraction Completed ===")

if __name__ == "__main__":
    main()
