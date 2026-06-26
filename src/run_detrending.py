import sys
import os

# Add parent directory to path to enable local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detrend import process_all_stars

def main():
    print("=== Phase 3: Detrending Started ===")
    process_all_stars()
    print("=== Phase 3: Detrending Completed ===")

if __name__ == "__main__":
    main()
