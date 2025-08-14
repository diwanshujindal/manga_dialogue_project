# Manga Dialogue Parser & Cleaner (VS Code Project)

This tool extracts dialogue from manga/comic pages and produces a **clean image** with the dialogue removed using inpainting.

## Features
- OCR using Tesseract via `pytesseract`
- Robust text-region detection (morphological ops + OCR boxes)
- Inpainting to remove the text:
  - **OpenCV** fast inpainting (default)
  - **Stable Diffusion inpainting** (optional) via `diffusers` for higher quality

## Quick Start (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# (Optional) Install Tesseract if not installed
# macOS (brew): brew install tesseract
# Ubuntu: sudo apt-get update && sudo apt-get install -y tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

Place an image in `data/samples/` (e.g., `page.jpg`) and run:
```bash
python -m src.main --image data/samples/page.jpg --mode both --lang eng --method opencv --outdir outputs
```

Outputs:
- `outputs/page_text.txt` — extracted dialogue
- `outputs/page_mask.png` — mask used for inpainting
- `outputs/page_clean.png` — image with dialogue removed

## Optional: High-Quality Inpainting (Stable Diffusion)
Install extra deps:
```bash
pip install torch torchvision diffusers transformers accelerate --upgrade
```
Then run:
```bash
python -m src.main --image data/samples/page.jpg --mode both --method sd --sd-model "stabilityai/stable-diffusion-2-inpainting"
```
> Note: SD inpainting is compute-heavy (prefer GPU).

## VS Code
- Debug configs under `.vscode/launch.json`
- Recommended to select `.venv` as the Python interpreter.

## CLI
```
python -m src.main --image <path> --mode [extract|clean|both] --method [opencv|sd]
                   [--lang eng] [--binary-thresh 200] [--outdir outputs]
                   [--sd-model stabilityai/stable-diffusion-2-inpainting]
```
