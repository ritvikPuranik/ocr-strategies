#!/usr/bin/env python3
"""
Approach 2: OCR + Layout (PaddleOCR with PP-StructureV3)

Uses PaddleOCR 3.x with layout detection for structured extraction.
In PaddleOCR 3.x, table/structure detection is integrated via PaddleOCR.predict()
Expected: Preserved table rows/columns, structured output.

Usage:
    python 02_ocr_layout.py

Requirements:
    pip install "paddleocr[doc-parser]"
    or: pip install paddlepaddle paddleocr
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    PADDLE_AVAILABLE = False
    print(f"Warning: Failed to import PaddleOCR: {e}")
    print("Try: pip install paddlepaddle paddleocr")

from config import get_image_paths, OUTPUT_DIR


def run_paddleocr(image_path: Path, ocr_engine) -> dict:
    """Run PaddleOCR 3.x on an image - returns OCR result"""
    # PaddleOCR 3.x uses predict() instead of ocr()
    result = ocr_engine.predict(input=str(image_path))
    
    words = []
    full_text_lines = []
    
    # Result structure: list of recognition results
    if hasattr(result, '__iter__'):
        for item in result:
            if hasattr(item, 'text'):
                text = item.text
                confidence = getattr(item, 'score', 0)
                bbox = getattr(item, 'bbox', {})
            else:
                continue
            
            words.append({
                "text": text,
                "conf": confidence,
                "bbox": bbox if isinstance(bbox, dict) else {"points": bbox}
            })
            full_text_lines.append(text)
    
    return {
        "words": words,
        "text": "\n".join(full_text_lines),
        "line_count": len(full_text_lines)
    }


def run_paddleocr_layout(image_path: Path, ocr_engine) -> dict:
    """Run PaddleOCR with layout detection (PP-StructureV3 mode)"""
    # In PaddleOCR 3.x, use CLI command for structure detection
    # Alternative: use subprocess to call the CLI if needed
    try:
        # Try using the predict method with layout parameters
        import os
        os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        
        # Use subprocess to call paddleocr CLI with structure detection
        cmd = [
            "paddleocr", "pp_structurev3", 
            "-i", str(image_path),
            "--use_doc_orientation_classify", "False",
            "--use_doc_unwarping", "False"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            output_str = result.stdout
            # Parse output - it returns JSON-like structure
            try:
                layout_result = json.loads(output_str)
            except json.JSONDecodeError:
                layout_result = {"raw": output_str}
        else:
            layout_result = {"error": result.stderr}
    except Exception as e:
        layout_result = {"error": str(e)}
    
    return layout_result


def main():
    print("=" * 60)
    print("Approach 2: OCR + Layout (PaddleOCR 3.x with PP-StructureV3)")
    print("=" * 60)
    
    if not PADDLE_AVAILABLE:
        print("\n✗ PaddleOCR not installed!")
        print("  Install with: pip install paddlepaddle paddleocr")
        return
    
    output_dir = OUTPUT_DIR / "02_ocr_layout"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OCR engine
    print("\nInitializing PaddleOCR...")
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    
    results = []
    
    for image_path in get_image_paths():
        print(f"\nProcessing: {image_path.name}")
        
        try:
            # Run OCR
            ocr_result = run_paddleocr(image_path, ocr_engine)
            print(f"  OCR: {len(ocr_result['words'])} words detected")
            
            # Run layout detection via CLI
            layout_result = run_paddleocr_layout(image_path, ocr_engine)
            if "error" not in layout_result:
                print(f"  Layout: Detected structure")
            else:
                print(f"  Layout: {layout_result.get('error', 'Error running structure detection')}")
            
            result = {
                "image": image_path.name,
                "approach": "ocr_layout",
                "model": "paddleocr_v3",
                "ocr": ocr_result,
                "layout": layout_result,
                "timestamp": datetime.now().isoformat()
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
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "approach": "02_ocr_layout",
            "model": "paddleocr_v3",
            "total_images": len(results),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
