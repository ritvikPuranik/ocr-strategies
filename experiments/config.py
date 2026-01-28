"""
Shared configuration for OCR experiments
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
IMAGE_DIR = PROJECT_ROOT / "archive" / "batch_1" / "batch_1" / "batch1_1"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs"

# Images for experiment (first 5)
EXPERIMENT_IMAGES = [
    "batch1-0001.jpg",
    # "batch1-0002.jpg",
    # "batch1-0003.jpg",
    # "batch1-0004.jpg",
    # "batch1-0005.jpg",
]

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_image_paths():
    """Return list of full paths to experiment images"""
    return [IMAGE_DIR / img for img in EXPERIMENT_IMAGES]
