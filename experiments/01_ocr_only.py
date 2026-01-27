#!/usr/bin/env python3
"""
Approach 1: OCR-only (Tesseract)

Uses Tesseract OCR directly without any layout detection.
Expected: Correct characters, but broken layout for tables/columns.

Usage:
    python 01_ocr_only.py

Or with Docker (if pytesseract not installed locally):
    See run_docker.sh in outputs folder
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Try pytesseract first, fall back to Docker
USE_DOCKER = True  # Force Docker for consistent results across platforms
try:
    import pytesseract
    from PIL import Image
    # Uncomment below to use pytesseract if tesseract binary is available locally
    # USE_DOCKER = False
except ImportError:
    pass

from config import get_image_paths, OUTPUT_DIR, IMAGE_DIR


def run_tesseract_docker(image_path: Path) -> str:
    """Run Tesseract via Docker container"""
    # Mount the image directory
    mount_path = image_path.parent
    image_name = image_path.name
    
    cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "-v", f"{mount_path}:/images",
        "tesseractshadow/tesseract4re:latest",
        "tesseract", f"/images/{image_name}", "stdout"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Tesseract failed: {result.stderr}")
    return result.stdout


def run_tesseract_local(image_path: Path) -> str:
    """Run Tesseract via pytesseract"""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


def run_tesseract_with_boxes(image_path: Path) -> dict:
    """Run Tesseract and get text with bounding boxes (for later LLM grounding)"""
    if USE_DOCKER:
        # Docker version - get TSV output
        mount_path = image_path.parent
        image_name = image_path.name
        
        cmd = [
            "docker", "run", "--rm",
            "--platform", "linux/amd64",
            "-v", f"{mount_path}:/images",
            "tesseractshadow/tesseract4re:latest",
            "tesseract", f"/images/{image_name}", "stdout", "tsv"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Tesseract failed: {result.stderr}")
        
        # Parse TSV to structured format
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            return {"words": [], "text": ""}
        
        headers = lines[0].split('\t')
        words = []
        for line in lines[1:]:
            parts = line.split('\t')
            if len(parts) >= 12 and parts[11].strip():  # Has text
                words.append({
                    "text": parts[11],
                    "conf": float(parts[10]) if parts[10] != '-1' else 0,
                    "bbox": {
                        "left": int(parts[6]),
                        "top": int(parts[7]),
                        "width": int(parts[8]),
                        "height": int(parts[9])
                    }
                })
        
        full_text = " ".join(w["text"] for w in words)
        return {"words": words, "text": full_text}
    else:
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        words = []
        for i, text in enumerate(data['text']):
            if text.strip():
                words.append({
                    "text": text,
                    "conf": data['conf'][i],
                    "bbox": {
                        "left": data['left'][i],
                        "top": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i]
                    }
                })
        
        full_text = " ".join(w["text"] for w in words)
        return {"words": words, "text": full_text}


def main():
    print("=" * 60)
    print("Approach 1: OCR-only (Tesseract)")
    print(f"Using: {'Docker' if USE_DOCKER else 'pytesseract (local)'}")
    print("=" * 60)
    
    output_dir = OUTPUT_DIR / "01_ocr_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for image_path in get_image_paths():
        print(f"\nProcessing: {image_path.name}")
        
        try:
            # Get plain text
            if USE_DOCKER:
                text = run_tesseract_docker(image_path)
            else:
                text = run_tesseract_local(image_path)
            
            # Get text with boxes (for comparison/later use)
            data_with_boxes = run_tesseract_with_boxes(image_path)
            
            result = {
                "image": image_path.name,
                "approach": "ocr_only",
                "model": "tesseract4",
                "text": text,
                "words_with_boxes": data_with_boxes["words"],
                "word_count": len(data_with_boxes["words"]),
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            # Save individual result
            out_file = output_dir / f"{image_path.stem}_ocr.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Also save plain text for easy viewing
            txt_file = output_dir / f"{image_path.stem}_ocr.txt"
            with open(txt_file, 'w') as f:
                f.write(text)
            
            print(f"  ✓ Extracted {len(data_with_boxes['words'])} words")
            print(f"  → Saved to: {out_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "image": image_path.name,
                "error": str(e)
            })
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "approach": "01_ocr_only",
            "model": "tesseract4",
            "total_images": len(results),
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"  Summary: {summary_file}")


if __name__ == "__main__":
    main()
