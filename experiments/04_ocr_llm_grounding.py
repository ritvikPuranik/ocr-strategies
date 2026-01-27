#!/usr/bin/env python3
"""
Approach 4: OCR + LLM with Grounding

Uses Tesseract for OCR (text + bounding boxes), then passes the grounded text
to an LLM (GPT-4o or Claude) with a schema prompt for structured extraction.

Usage:
    # With OpenAI
    export OPENAI_API_KEY="your-key"
    python 04_ocr_llm_grounding.py

    # With Anthropic Claude
    export ANTHROPIC_API_KEY="your-key"  
    python 04_ocr_llm_grounding.py --provider anthropic

Requirements:
    pip install openai anthropic pytesseract Pillow
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import os

# Import OCR function from approach 1
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("ocr_only", Path(__file__).parent / "01_ocr_only.py")
ocr_module = module_from_spec(spec)
spec.loader.exec_module(ocr_module)
run_tesseract_with_boxes = ocr_module.run_tesseract_with_boxes

from config import get_image_paths, OUTPUT_DIR


# Invoice extraction schema
INVOICE_SCHEMA = {
    "vendor": {
        "name": "string",
        "address": "string",
        "phone": "string",
        "email": "string"
    },
    "invoice_number": "string",
    "invoice_date": "string (YYYY-MM-DD)",
    "due_date": "string (YYYY-MM-DD)",
    "customer": {
        "name": "string",
        "address": "string"
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
    "total": "number",
    "payment_terms": "string"
}


def format_ocr_for_llm(ocr_data: dict) -> str:
    """Format OCR output with positions for LLM consumption"""
    lines = []
    lines.append("OCR OUTPUT WITH POSITIONS:")
    lines.append("=" * 50)
    
    # Group words by approximate line (based on Y position)
    words = ocr_data["words"]
    if not words:
        return "No text detected"
    
    # Sort by top position, then left
    sorted_words = sorted(words, key=lambda w: (w["bbox"]["top"], w["bbox"]["left"]))
    
    # Group into lines (words within 20px vertically)
    current_line = []
    current_y = sorted_words[0]["bbox"]["top"] if sorted_words else 0
    line_num = 1
    
    for word in sorted_words:
        word_y = word["bbox"]["top"]
        if abs(word_y - current_y) > 20:  # New line
            if current_line:
                # Sort line by X position
                current_line.sort(key=lambda w: w["bbox"]["left"])
                line_text = " ".join(w["text"] for w in current_line)
                x_range = f"x:{current_line[0]['bbox']['left']}-{current_line[-1]['bbox']['left'] + current_line[-1]['bbox']['width']}"
                lines.append(f"L{line_num:03d} (y:{current_y:4d}, {x_range}): {line_text}")
                line_num += 1
            current_line = [word]
            current_y = word_y
        else:
            current_line.append(word)
    
    # Don't forget last line
    if current_line:
        current_line.sort(key=lambda w: w["bbox"]["left"])
        line_text = " ".join(w["text"] for w in current_line)
        x_range = f"x:{current_line[0]['bbox']['left']}-{current_line[-1]['bbox']['left'] + current_line[-1]['bbox']['width']}"
        lines.append(f"L{line_num:03d} (y:{current_y:4d}, {x_range}): {line_text}")
    
    return "\n".join(lines)


def build_prompt(ocr_formatted: str) -> str:
    """Build the extraction prompt for the LLM"""
    return f"""You are an expert document parser. Extract structured data from the following OCR output of an invoice/receipt.

The OCR output includes line numbers and approximate positions (y = vertical position, x = horizontal range).
Use the positions to understand the document layout, especially for tables where columns may have been mixed.

{ocr_formatted}

Extract the information into this JSON schema:
{json.dumps(INVOICE_SCHEMA, indent=2)}

Rules:
1. If a field is not found, use null
2. For dates, convert to YYYY-MM-DD format
3. For numbers, extract numeric values only (remove currency symbols)
4. For line_items, carefully reconstruct table rows using position information
5. Validate that line item totals = quantity × unit_price
6. Validate that subtotal + tax ≈ total

Respond with ONLY the JSON object, no markdown code blocks or explanation."""


def call_openai(prompt: str, api_key: str) -> dict:
    """Call OpenAI API"""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise document parser that extracts structured data from OCR output."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2000
    )
    
    content = response.choices[0].message.content
    
    # Parse JSON from response
    try:
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_response": content, "parse_error": True}


def call_anthropic(prompt: str, api_key: str) -> dict:
    """Call Anthropic Claude API"""
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    content = response.content[0].text
    
    # Parse JSON from response
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_response": content, "parse_error": True}


def main():
    parser = argparse.ArgumentParser(description="OCR + LLM Grounding approach")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                        help="LLM provider to use")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Approach 4: OCR + LLM with Grounding")
    print(f"Provider: {args.provider}")
    print("=" * 60)
    
    # Check API key
    if args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("\n✗ OPENAI_API_KEY not set!")
            print("  Run: export OPENAI_API_KEY='your-key'")
            return
        llm_call = lambda p: call_openai(p, api_key)
        model_name = "gpt-4o"
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("\n✗ ANTHROPIC_API_KEY not set!")
            print("  Run: export ANTHROPIC_API_KEY='your-key'")
            return
        llm_call = lambda p: call_anthropic(p, api_key)
        model_name = "claude-sonnet-4-20250514"
    
    output_dir = OUTPUT_DIR / "04_ocr_llm_grounding"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for image_path in get_image_paths():
        print(f"\nProcessing: {image_path.name}")
        
        try:
            # Step 1: OCR with bounding boxes
            print("  Step 1: Running OCR...")
            ocr_data = run_tesseract_with_boxes(image_path)
            print(f"    → {len(ocr_data['words'])} words extracted")
            
            # Step 2: Format for LLM
            print("  Step 2: Formatting for LLM...")
            ocr_formatted = format_ocr_for_llm(ocr_data)
            
            # Step 3: Call LLM
            print(f"  Step 3: Calling {model_name}...")
            prompt = build_prompt(ocr_formatted)
            extracted = llm_call(prompt)
            
            result = {
                "image": image_path.name,
                "approach": "ocr_llm_grounding",
                "ocr_model": "tesseract4",
                "llm_model": model_name,
                "ocr_word_count": len(ocr_data["words"]),
                "extracted": extracted,
                "ocr_formatted": ocr_formatted,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            # Save individual result
            out_file = output_dir / f"{image_path.stem}_llm.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Check for parse errors
            if isinstance(extracted, dict) and extracted.get("parse_error"):
                print(f"  ⚠ LLM response could not be parsed as JSON")
            else:
                print(f"  ✓ Extracted fields: {list(extracted.keys()) if isinstance(extracted, dict) else 'error'}")
            
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
            "approach": "04_ocr_llm_grounding",
            "ocr_model": "tesseract4",
            "llm_model": model_name,
            "total_images": len(results),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
