import sys
import os

# Add parent directory to path to enable local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.period_search import run_period_search_all

def main():
    print("=== Phase 4: BLS Period Search Started ===")
    run_period_search_all()
    print("=== Phase 4: BLS Period Search Completed ===")

if __name__ == "__main__":
    main()
