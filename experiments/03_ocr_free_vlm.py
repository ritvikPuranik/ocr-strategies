#!/usr/bin/env python3
"""
Approach 3: OCR-free Vision Model (Donut)

Uses Donut (Document Understanding Transformer) - an end-to-end vision model
that extracts structured data directly from images without separate OCR.

Model: naver-clova-ix/donut-base-finetuned-cord-v2 (trained on receipts/invoices)

Usage:
    python 03_ocr_free_vlm.py

Requirements:
    pip install transformers torch sentencepiece
"""
import json
import re
from pathlib import Path
from datetime import datetime

try:
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not installed. Run: pip install transformers torch sentencepiece")

from config import get_image_paths, OUTPUT_DIR


# Model options - choose based on your use case
MODELS = {
    "cord": "naver-clova-ix/donut-base-finetuned-cord-v2",  # Receipts/invoices
    "docvqa": "naver-clova-ix/donut-base-finetuned-docvqa",  # Document QA
    "rvlcdip": "naver-clova-ix/donut-base-finetuned-rvlcdip",  # Document classification
}

# Using CORD model for invoice/receipt extraction
MODEL_NAME = MODELS["cord"]


def load_model():
    """Load Donut model and processor"""
    print(f"Loading model: {MODEL_NAME}")
    print("(This may take a few minutes on first run...)")
    
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    return processor, model, device


def run_donut(image_path: Path, processor, model, device) -> dict:
    """Run Donut model on an image"""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare input
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Generate output
    task_prompt = "<s_cord-v2>"  # CORD task prompt
    decoder_input_ids = processor.tokenizer(
        task_prompt, 
        add_special_tokens=False, 
        return_tensors="pt"
    ).input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,  # Greedy decoding for speed
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    
    # Decode output
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove task prompt
    
    # Parse JSON from output
    try:
        # Donut outputs JSON-like structure
        parsed = processor.token2json(sequence)
    except Exception:
        parsed = {"raw_output": sequence}
    
    return {
        "raw_sequence": sequence,
        "parsed": parsed
    }


def main():
    print("=" * 60)
    print("Approach 3: OCR-free Vision Model (Donut)")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n✗ Required packages not installed!")
        print("  Install with: pip install transformers torch sentencepiece")
        return
    
    output_dir = OUTPUT_DIR / "03_ocr_free_vlm"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        processor, model, device = load_model()
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print("  Make sure you have enough memory and internet connection.")
        return
    
    results = []
    
    for image_path in get_image_paths():
        print(f"\nProcessing: {image_path.name}")
        
        try:
            # Run Donut
            donut_result = run_donut(image_path, processor, model, device)
            
            result = {
                "image": image_path.name,
                "approach": "ocr_free_vlm",
                "model": MODEL_NAME,
                "output": donut_result["parsed"],
                "raw_sequence": donut_result["raw_sequence"],
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            # Save individual result
            out_file = output_dir / f"{image_path.stem}_vlm.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Extracted fields: {list(donut_result['parsed'].keys()) if isinstance(donut_result['parsed'], dict) else 'raw'}")
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
            "model": MODEL_NAME,
            "total_images": len(results),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
