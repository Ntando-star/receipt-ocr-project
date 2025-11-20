# ocr_tesseract.py
import os
import time
import json
import argparse
from pathlib import Path
from PIL import Image
import pytesseract

def run_tesseract_on_file(image_path):
    """Run Tesseract OCR on a single image and return text."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

def process_file(image_path, output_dir):
    """Run OCR, save output, and return text with runtime."""
    start_time = time.time()

    ocr_text = run_tesseract_on_file(image_path)

    fname = Path(image_path).stem + ".txt"
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    elapsed_time = time.time() - start_time
    print(f"✅ OCR done: {out_path} (Time: {elapsed_time:.2f}s)")

    return fname, ocr_text

def main(input_dir: str, output_dir: str, save_json: bool = True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = output_dir / "results.json"
    results = {}

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    for fname in image_files:
        image_path = input_dir / fname
        key, text = process_file(str(image_path), str(output_dir))
        # store even empty outputs to keep mapping consistent
        results[key] = text

    if save_json:
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n📄 All OCR results saved to {results_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tesseract OCR on a folder of images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save OCR .txt files and results.json")
    parser.add_argument("--no_json", action="store_true", help="Do not save consolidated results.json")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, save_json=not args.no_json)
