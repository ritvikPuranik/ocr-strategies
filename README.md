# ocr-strategies
A simple pipeline to test various OCR strategies

A typical OCR pipeline, and the steps associated with it
| Stage            | Input          | Output         | Common Failures        |
| ---------------- | -------------- | -------------- | ---------------------- |
| Preprocessing    | Raw image      | Cleaned image  | Blur, skew, noise      |
| Text Detection   | Image          | Bounding boxes | Missed small text      |
| Text Recognition | Crop           | Text           | `0` vs `O`, `1` vs `l` |
| Ordering         | Boxes          | Sequence       | Columns mixed          |
| Layout           | Tokens + boxes | Groups         | Tables collapse        |
| Semantics        | Layout         | Fields         | Wrong totals           |


### Expt 1 - Direct OCR through CLI - No preprocessing/ Layout detection
Image - Clean invoice image, with table and columns
Model - tesseract4re:latest (running as a docker container)
Observations -
- Text was detected correctly
- Layout was messed up - lost the meaning
    - OCR processed it as left and right halves of the document - table in between was broken up at a column
- Need Layout Context before feeding to OCR engine / Vision based transformers to store context of the layout

## Text Extraction vs Document Understanding

Text Extraction: Converts pixels to characters. Input: image; Output: text tokens with confidences and often boxes. Tools: Tesseract, PaddleOCR. Failures: character confusions, missed small text, wrong reading order.
Document Understanding: Assigns meaning and structure to extracted text. Input: tokens + boxes; Output: fields, tables, hierarchies, relations. Tools: layout analyzers, table detectors, key-value extractors, LLM parsers. Failures: wrong grouping, broken tables, mis-assigned fields.
When to use: Extraction for searchable text or simple strings; understanding for invoices, forms, tables, multi-column docs where layout and semantics matter.

## OCR Engine vs Layout Model vs LLM Parser

OCR Engine: Detects text regions and recognizes characters/words. Strengths: speed, accuracy on clean text. Limits: weak layout/semantics, can mix columns or break tables.
Layout Model: Detects and organizes structural elements (pages, columns, headings, paragraphs, tables, cells). Examples: models trained on PubLayNet; CV detectors for table grids; reading-order estimators. Strengths: preserves structure and grouping; Limits: does not “read” text content by itself.
LLM Parser: Consumes text (optionally with positions) and maps to schemas (e.g., JSON of fields, table rows). Strengths: semantic reasoning, normalization, fuzzy matching; Limits: depends on correct reading order and sufficient layout cues; may hallucinate without grounding.
Interplay: Typical pipeline is OCR → Layout → LLM. OCR provides tokens, layout preserves structure and order, LLM maps content to fields with validation.

## Vision-based vs Text-based Approaches

Vision-based: Operate on pixels/boxes. Examples: OCR, object detection for tables, OCR-free models (e.g., encoder–decoder vision transformers). Pros: robust to layout complexity, can infer structure directly; Cons: require GPU, may need fine-tuning and good annotations.
Text-based: Operate on extracted text (and optionally coordinates). Examples: regex/rules, graph clustering of boxes, LLM prompts over text with TSV/JSON of positions. Pros: simple, cheap, easy to iterate; Cons: fragile to reading-order errors, loses layout cues if coordinates are missing.
Hybrid (common in practice): Vision to detect blocks/tables and establish reading order; text methods/LLMs for semantic extraction and normalization.
Choosing in practice (for invoices/tables like your Expt 1)

OCR-only: Fast baseline; expect correct characters but broken layout (columns/tables).
OCR + Layout: Add table detection (CV grid detection or table-cell detector), column detection, then read order per block → preserves table rows/columns.
OCR-free vision model: Use a document VLM to output structured fields; good if you can fine-tune and accept heavier inference.
OCR + LLM with grounding: Feed tokens plus bounding boxes/line grouping to the LLM; constrain with schemas and validations to reduce hallucinations.

## Evaluation focus

Extraction metrics: character/word accuracy, confidence, recall on small text.
Layout metrics: block/column/table detection F1, reading-order correctness, table cell assignment accuracy.
Semantics: field-level precision/recall, totals validation, cross-field consistency.

---

## Expt 2 - Comparing 4 OCR Approaches

Testing 4 different approaches on invoice images from `batch1_1` (first 5 images).

### Approaches

| # | Approach | Model/Tool | Expected Outcome |
|---|----------|------------|------------------|
| 1 | OCR-only | Tesseract 4 (Docker) | Correct text, broken layout |
| 2 | OCR + Layout | PaddleOCR + PP-Structure | Preserved tables/columns |
| 3 | OCR-free VLM | Donut (CORD-v2) | End-to-end structured output |
| 4 | OCR + LLM Grounding | Tesseract + GPT-4o/Claude | Semantic field extraction |

### Running the Experiments

```bash
cd experiments

# Install dependencies
pip install -r requirements.txt

# Run all approaches
python run_all.py

# Or run individually
python 01_ocr_only.py           # Uses Docker Tesseract
python 02_ocr_layout.py         # Needs paddleocr installed
python 03_ocr_free_vlm.py       # Needs transformers, torch
python 04_ocr_llm_grounding.py  # Needs OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### Output Structure

```
experiments/outputs/
├── 01_ocr_only/
│   ├── batch1-0001_ocr.txt      # Plain text output
│   ├── batch1-0001_ocr.json     # Text + bounding boxes
│   └── summary.json
├── 02_ocr_layout/
│   ├── batch1-0001_layout.json  # OCR + structure
│   ├── batch1-0001_tables.html  # Detected tables as HTML
│   └── summary.json
├── 03_ocr_free_vlm/
│   ├── batch1-0001_vlm.json     # VLM extracted fields
│   └── summary.json
└── 04_ocr_llm_grounding/
    ├── batch1-0001_llm.json     # LLM parsed JSON
    └── summary.json
```

### Observations

| Approach | Text Accuracy | Layout Preserved | Semantic Fields | Speed | Notes |
|----------|--------------|------------------|-----------------|-------|-------|
| 1. OCR-only | TBD | TBD | N/A | Fast | Baseline |
| 2. OCR + Layout | TBD | TBD | TBD | Medium | |
| 3. OCR-free VLM | TBD | TBD | TBD | Slow | GPU recommended |
| 4. OCR + LLM | TBD | TBD | TBD | Slow | API costs |