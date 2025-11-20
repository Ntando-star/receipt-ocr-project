import os
import time
from pathlib import Path
import json
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np

# Directories
INPUT_DIR = "data/raw/left_overs"
OUTPUT_DIR = "data/ocr_output_paddle_val"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(OUTPUT_DIR, "results.json")

# 🚀 Updated PaddleOCR initialization for latest API
ocr = PaddleOCR(
    lang='en',
    use_textline_orientation=False,
)

MAX_SIDE = 1500  # Resize to improve speed

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return np.array(img)

def run_paddleocr_on_file(image_path):
    img = preprocess_image(image_path)

    result = ocr.predict(img)  # <-- Correct API

    lines = []
    for block in result:
        if isinstance(block, dict) and "rec_texts" in block:
            lines.extend(block["rec_texts"])  # text list

    return "\n".join(lines)

def process_file(image_path):
    start = time.time()
    text = run_paddleocr_on_file(image_path)

    out_name = Path(image_path).stem + ".txt"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Done: {out_path} ({time.time() - start:.2f}s)")
    return out_name, text

if __name__ == "__main__":
    results = {}

    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            key, text = process_file(os.path.join(INPUT_DIR, fname))
            results[key] = text

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📄 All OCR results saved to {RESULTS_JSON}\n")
