import sys
import os

# Add parent directory to path to enable local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifier import train_and_evaluate

def main():
    print("=== Phase 7 & 8: Classification & Scoring Started ===")
    train_and_evaluate()
    print("=== Phase 7 & 8: Classification & Scoring Completed ===")

if __name__ == "__main__":
    main()
