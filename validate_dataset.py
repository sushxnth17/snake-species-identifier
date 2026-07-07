"""
CLI tool for validating the snake dataset.
Wraps the reusable DatasetValidator module.
"""

import sys
from ml.dataset_validator import DatasetValidator

def validate_dataset(dataset_dir: str = "dataset"):
    """
    Validates the dataset directory and prints a report.
    """
    try:
        validator = DatasetValidator(dataset_dir)
        results = validator.validate()
        validator.print_summary(results)
        
        # If there are corrupted images, exit with non-zero code to signal failure to CI/CD pipelines
        if results["corrupted_images"]:
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Allow custom dataset folder as first command-line argument
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    validate_dataset(target_dir)
