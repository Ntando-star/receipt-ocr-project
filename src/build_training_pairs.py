import os
import json

# Choose OCR output folder: Tesseract or Paddle
ocr_folder = "data/ocr_output_tesseract"        # change to "data/ocr_output_paddle" if needed
gt_folder = "data/raw/ground_truth"
output_folder = "data/training_pairs"

os.makedirs(output_folder, exist_ok=True)

ocr_files = [f for f in os.listdir(ocr_folder) if f.endswith(".txt")]

for ocr_file in ocr_files:
    base = os.path.splitext(ocr_file)[0]
    ocr_path = os.path.join(ocr_folder, ocr_file)
    gt_path = os.path.join(gt_folder, f"{base}.txt")
    out_path = os.path.join(output_folder, f"{base}.json")

    if not os.path.exists(gt_path):
        print(f"⚠️ Ground truth missing for {base}, skipping")
        continue

    with open(ocr_path, "r") as f:
        ocr_text = f.read().strip()

    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    data = {
        "ocr_output_tesseract": ocr_text,
        "ground_truth": ground_truth
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Created training pair: {base}")
