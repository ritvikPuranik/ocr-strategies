#!/usr/bin/env python3
"""
Master runner for all OCR experiments

Usage:
    python run_all.py              # Run all approaches
    python run_all.py --approach 1 # Run only approach 1
    python run_all.py --approach 1 2 3  # Run approaches 1, 2, 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = {
    1: ("01_ocr_only.py", "OCR-only (Tesseract)"),
    2: ("02_ocr_layout.py", "OCR + Layout (PaddleOCR)"),
    3: ("03_ocr_free_vlm.py", "OCR-free VLM (Donut)"),
    4: ("04_ocr_llm_grounding.py", "OCR + LLM Grounding"),
}


def run_experiment(num: int, script: str, name: str):
    """Run a single experiment"""
    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT {num}: {name}")
    print("=" * 70)
    
    script_path = Path(__file__).parent / script
    result = subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent)
    
    if result.returncode != 0:
        print(f"\n⚠ Experiment {num} finished with errors")
    else:
        print(f"\n✓ Experiment {num} completed successfully")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run OCR experiments")
    parser.add_argument("--approach", "-a", type=int, nargs="+", choices=[1, 2, 3, 4],
                        help="Which approach(es) to run (1-4). Default: all")
    args = parser.parse_args()
    
    approaches = args.approach if args.approach else [1, 2, 3, 4]
    
    print("=" * 70)
    print("OCR EXPERIMENTS RUNNER")
    print("=" * 70)
    print(f"\nRunning approaches: {approaches}")
    print("\nApproaches available:")
    for num, (script, name) in EXPERIMENTS.items():
        marker = "→" if num in approaches else " "
        print(f"  {marker} {num}. {name}")
    
    results = {}
    for num in approaches:
        script, name = EXPERIMENTS[num]
        results[num] = run_experiment(num, script, name)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for num in approaches:
        script, name = EXPERIMENTS[num]
        status = "✓ Success" if results[num] == 0 else "✗ Failed"
        print(f"  {num}. {name}: {status}")
    
    print("\nOutputs saved to: experiments/outputs/")
    print("  01_ocr_only/      - Plain text + word boxes")
    print("  02_ocr_layout/    - Structured layout + tables")
    print("  03_ocr_free_vlm/  - VLM extracted fields")
    print("  04_ocr_llm_grounding/ - LLM parsed JSON")


if __name__ == "__main__":
    main()
