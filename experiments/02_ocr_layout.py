#!/usr/bin/env python3
"""
Approach 2: OCR + Layout (PaddleOCR 3.x)

Two modes:
  - lite (default): OCR only, faster, lower memory (good baseline on CPU/M1)
  - structure: invokes PP-StructureV3 via CLI for layout/table detection (heavier)

Usage:
    python 02_ocr_layout.py --mode lite         # fast baseline
    python 02_ocr_layout.py --mode structure    # full layout (slower)

Recommendations for M1/CPU:
  - keep --mode lite for throughput
  - downscale large pages (default max_dim=1600)
  - avoid re-initializing models per request; reuse the engine

Requirements:
    pip install "paddleocr[doc-parser]"    # for structure mode
    or: pip install paddlepaddle paddleocr # for lite mode only
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from PIL import Image

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    PADDLE_AVAILABLE = False
    print(f"Warning: Failed to import PaddleOCR: {e}")
    print("Try: pip install paddlepaddle paddleocr")

from config import get_image_paths, OUTPUT_DIR


def downscale_image(image_path: Path, max_dim: int, work_dir: Path) -> Path:
    """Downscale large images to max_dim on the long side. Returns path to downscaled image (or original if small)."""
    img = Image.open(image_path)
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_dim:
        return image_path  # no resize needed

    scale = max_dim / float(long_side)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{image_path.stem}_resized.jpg"
    img.save(out_path, quality=90)
    return out_path


def run_paddleocr(image_path: Path, ocr_engine) -> dict:
    """Run PaddleOCR 3.x on an image - returns OCR result"""
    result = ocr_engine.predict(input=str(image_path))

    words = []
    full_text_lines = []

    for item in result:
        try:
            text = item.text if hasattr(item, 'text') else str(item)
            score = item.score if hasattr(item, 'score') else 1.0
            bbox = item.bbox if hasattr(item, 'bbox') else {}

            words.append({
                "text": text,
                "conf": score,
                "bbox": bbox
            })
            full_text_lines.append(text)
        except Exception as e:
            print(f"    Warning: Could not parse item {item}: {e}")
            continue

    return {
        "words": words,
        "text": "\n".join(full_text_lines),
        "line_count": len(full_text_lines)
    }


def run_paddleocr_layout(image_path: Path, timeout_s: int) -> dict:
    """Run PaddleOCR PP-StructureV3 via CLI. Heavy; use only in structure mode."""
    os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

    print("    Running PP-StructureV3 (heavy). First run may download large models...")
    cmd = [
        "paddleocr", "pp_structurev3",
        "-i", str(image_path),
        "--use_doc_orientation_classify", "False",
        "--use_doc_unwarping", "False"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if result.returncode == 0:
            output_str = result.stdout
            try:
                layout_result = json.loads(output_str)
            except json.JSONDecodeError:
                layout_result = {"raw": output_str}
        else:
            error_msg = result.stderr
            if "DependencyError" in error_msg or "additional dependencies" in error_msg:
                return {
                    "error": "Missing dependencies",
                    "fix": "Run: pip install \"paddleocr[doc-parser]\" or pip install \"paddleocr[all]\"",
                    "details": error_msg
                }
            return {"error": error_msg, "stdout": result.stdout}
    except subprocess.TimeoutExpired:
        return {
            "error": f"Timeout - structure detection took too long (>{timeout_s}s)",
            "note": "First run downloads models; consider warming up or using lite mode."
        }
    except Exception as e:
        return {"error": str(e)}

    return layout_result


def parse_args():
    parser = argparse.ArgumentParser(description="PaddleOCR layout experiment")
    parser.add_argument("--mode", choices=["lite", "structure"], default="lite",
                        help="lite = OCR only (fast), structure = PP-StructureV3 (heavy)")
    parser.add_argument("--max-dim", type=int, default=1600,
                        help="Max long-side resolution before OCR/layout")
    parser.add_argument("--timeout", type=int, default=1000,
                        help="Timeout in seconds for structure mode")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads (for paddle backend)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Approach 2: OCR + Layout (mode={args.mode})")
    print("=" * 60)
    
    if not PADDLE_AVAILABLE:
        print("\n✗ PaddleOCR not installed!")
        print("  Install with: pip install paddlepaddle paddleocr")
        return
    
    # Set thread env for CPU (M1/CPU-only suggestion)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    output_dir = OUTPUT_DIR / "02_ocr_layout"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"

    print("\nInitializing PaddleOCR (lite OCR engine)...")
    ocr_engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    
    results = []
    
    for image_path in get_image_paths():
        print(f"\nProcessing: {image_path.name}")
        
        try:
            # Downscale to reduce latency / memory
            resized_path = downscale_image(image_path, max_dim=args.max_dim, work_dir=work_dir)
            if resized_path != image_path:
                print(f"  Downscaled to {resized_path.name}")
            
            # Run OCR (lite)
            ocr_result = run_paddleocr(resized_path, ocr_engine)
            print(f"  OCR: {len(ocr_result['words'])} words detected")
            
            # Layout step (optional)
            if args.mode == "structure":
                layout_result = run_paddleocr_layout(resized_path, timeout_s=args.timeout)
                if "error" not in layout_result:
                    print(f"  Layout: Detected structure")
                elif layout_result.get("error") == "Missing dependencies":
                    print(f"  Layout: ⚠ Missing dependencies")
                    print(f"    Fix: {layout_result.get('fix')}")
                else:
                    print(f"  Layout: Error - {layout_result.get('error', 'Unknown error')}")
            else:
                layout_result = {"note": "structure step skipped (lite mode)"}
                print("  Layout: skipped (lite mode)")
            
            result = {
                "image": image_path.name,
                "approach": "ocr_layout",
                "mode": args.mode,
                "model": "paddleocr_v3",
                "ocr": ocr_result,
                "layout": layout_result,
                "timestamp": datetime.now().isoformat(),
                "max_dim": args.max_dim
            }
            results.append(result)
            
            # Save individual result
            out_file = output_dir / f"{image_path.stem}_layout.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Save plain text
            txt_file = output_dir / f"{image_path.stem}_layout.txt"
            with open(txt_file, 'w') as f:
                f.write(ocr_result['text'])
            
            print(f"  ✓ Saved to: {out_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "image": image_path.name,
                "error": str(e)
            })
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "approach": "02_ocr_layout",
            "model": "paddleocr_v3",
            "mode": args.mode,
            "max_dim": args.max_dim,
            "total_images": len(results),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
