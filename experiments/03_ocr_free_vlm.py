#!/usr/bin/env python3
"""
Approach 3: OCR-free Vision Model (Ollama VLM)

Uses Ollama-served vision-language models that extract structured data 
directly from images without separate OCR.

Supported models (vision-capable):
  - llama3.2-vision:11b (recommended)
  - minicpm-v:8b (excellent for documents, lighter)
  - llava:7b / llava:13b

Usage:
    # First, pull a vision model in Ollama:
    ollama pull minicpm-v:8b
    
    # Then run:
    python 03_ocr_free_vlm.py --model minicpm-v:8b
    python 03_ocr_free_vlm.py --model llama3.2-vision:11b

Requirements:
    pip install ollama pillow
    brew install ollama  # if not installed
"""
import argparse
import json
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO

try:
    import ollama
    from PIL import Image
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not installed. Run: pip install ollama")

from config import get_image_paths, OUTPUT_DIR


# Invoice extraction prompt for VLM
EXTRACTION_PROMPT = """Analyze this document image and extract structured information in JSON format.

Extract the following fields (use null if not found):
{
  "seller": {
    "name": "string",
    "address": "string",
    "tax id": "string"
  },
  "invoice_number": "string",
  "invoice_date": "string (YYYY-MM-DD)",
  "due_date": "string (YYYY-MM-DD)",
  "client": {
    "name": "string",
    "address": "string",
    "tax id": "string"
  },
  "line_items": [
    {
      "description": "string",
      "quantity": "number",
      "unit_price": "number",
      "total": "number"
    }
  ],
  "subtotal": "number",
  "tax": "number",
  "total": "number"
}
Explain what you see in simple words before providing the JSON output."""
# Respond ONLY with valid JSON, no markdown formatting or explanation."""


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 for Ollama API"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def run_ollama_vlm(image_path: Path, model: str) -> dict:
    """Run Ollama vision model on an image"""
    try:
        # Prepare image
        img_b64 = image_to_base64(image_path)
        
        # Call Ollama with vision
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': EXTRACTION_PROMPT,
                'images': [img_b64]
            }]
        )
        
        content = response['message']['content']
        print(f"vlm response -> {content}")
        
        # Parse JSON from response
        try:
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            parsed = json.loads(content.strip())
            return {
                "parsed": parsed,
                "raw_response": content,
                "success": True
            }
        except json.JSONDecodeError as e:
            return {
                "raw_response": content,
                "parse_error": str(e),
                "success": False
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def parse_args():
    parser = argparse.ArgumentParser(description="OCR-free VLM via Ollama")
    parser.add_argument("--model", default="gemma3:4b",
                        help="Ollama vision model to use (must be vision-capable)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"Approach 3: OCR-free Vision Model (Ollama)")
    print(f"Model: {args.model}")
    print("=" * 60)
    
    if not OLLAMA_AVAILABLE:
        print("\n✗ Required packages not installed!")
        print("  Install with: pip install ollama pillow")
        return
    
    # Check if Ollama is running and model is available
    try:
        print("\nChecking Ollama connection...")
        models_response = ollama.list()
        
        # Debug: print response structure
        # print(f"DEBUG: Response type: {type(models_response)}")
        # print(f"DEBUG: Response: {models_response}")
        
        # Handle different response structures
        available_models = []
        if isinstance(models_response, dict):
            if 'models' in models_response:
                available_models = [m.get('name', m.get('model', '')) for m in models_response['models']]
        elif isinstance(models_response, list):
            available_models = [m.get('name', m.get('model', '')) for m in models_response]
        
        if available_models:
            print(f"✓ Found {len(available_models)} models in Ollama")
            if args.model not in available_models:
                print(f"\n⚠ Model '{args.model}' not found in Ollama!")
                print(f"  Available models: {', '.join(available_models[:5])}")
                print(f"\n  Pull it with: ollama pull {args.model}")
                print("\n  Recommended vision models:")
                print("    - gemma3:4b (multimodal, good for M1)")
                print("    - minicpm-v:8b (excellent for documents, lighter)")
                print("    - llama3.2-vision:11b (more capable, heavier)")
                return
            else:
                print(f"✓ Model '{args.model}' found")
        else:
            print("⚠ Could not parse Ollama model list, but continuing...")
            print(f"  Attempting to use '{args.model}' anyway...")
            print("  If it fails, run: ollama pull {args.model}")
            
    except Exception as e:
        print(f"⚠ Warning: Could not connect to Ollama: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        print(f"  Continuing anyway - inference will fail if model '{args.model}' isn't available")
        print("")
    
    output_dir = OUTPUT_DIR / "03_ocr_free_vlm"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for image_path in get_image_paths():
        print(f"\nProcessing: {image_path.name}")
        
        try:
            # Run VLM
            print(f"  Calling {args.model}...")
            vlm_result = run_ollama_vlm(image_path, args.model)
            
            result = {
                "image": image_path.name,
                "approach": "ocr_free_vlm",
                "model": args.model,
                "output": vlm_result.get("parsed", {}),
                "raw_response": vlm_result.get("raw_response", ""),
                "success": vlm_result.get("success", False),
                "timestamp": datetime.now().isoformat()
            }
            
            if vlm_result.get("error"):
                result["error"] = vlm_result["error"]
            if vlm_result.get("parse_error"):
                result["parse_error"] = vlm_result["parse_error"]
            
            results.append(result)
            
            # Save individual result
            out_file = output_dir / f"{image_path.stem}_vlm.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            if vlm_result.get("success"):
                fields = list(vlm_result["parsed"].keys()) if isinstance(vlm_result["parsed"], dict) else []
                print(f"  ✓ Extracted fields: {', '.join(fields) if fields else 'see raw response'}")
            else:
                print(f"  ⚠ Parse failed: {vlm_result.get('error') or vlm_result.get('parse_error', 'Unknown')}")
            
            print(f"  → Saved to: {out_file.name}")
            
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
            "approach": "03_ocr_free_vlm",
            "model": args.model,
            "total_images": len(results),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
