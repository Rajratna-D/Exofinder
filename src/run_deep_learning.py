import sys
import os

# Add parent directory to path to enable local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.deep_learning import train_and_evaluate_cnn

def main():
    print("=== Phase 9: NASA Astronet 1D-CNN Deep Learning Training Started ===")
    train_and_evaluate_cnn()
    print("=== Phase 9: NASA Astronet 1D-CNN Deep Learning Training Completed ===")

if __name__ == "__main__":
    main()
