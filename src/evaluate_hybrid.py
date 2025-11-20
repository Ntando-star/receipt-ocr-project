# src/evaluate_hybrid.py
import json
from pathlib import Path
from tabulate import tabulate
from src.hybrid_extractor import HybridExtractor

# Fields to evaluate
FIELDS = ["company", "address", "date", "total"]

# Directories
GT_DIR = Path("data/raw/ground_truth")
OCR_DIRS = {
    "tesseract": Path("data/ocr_output_tesseract_test"),
    "paddle": Path("data/ocr_output_paddle_test")
}
ENSEMBLE_MODEL_DIRS = {
    "tesseract": Path("data/models/ensemble_tesseract"),
    "paddle": Path("data/models/ensemble_paddle")
}

def load_test_data(ocr_dir: Path, field: str):
    """Load OCR text and ground truth for the test set."""
    data = []
    for gt_file in GT_DIR.glob("*.txt"):
        with open(gt_file, "r") as f:
            gt = json.load(f)
        ocr_file = ocr_dir / gt_file.name
        ocr_text = ocr_file.read_text() if ocr_file.exists() else ""
        data.append({"ocr_text": ocr_text, "ground_truth": gt.get(field, "")})
    return data

def string_similarity(a: str, b: str):
    """Compute simple similarity ratio."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

def evaluate_hybrid(hybrid_extractor: HybridExtractor, test_data: list):
    exact_matches = 0
    similarity_scores = []

    for sample in test_data:
        ocr_text = sample["ocr_text"]
        gt = sample["ground_truth"]

        result = hybrid_extractor.extract(ocr_text)
        pred = result["value"] or ""

        # Exact match
        if pred.strip() == gt.strip():
            exact_matches += 1

        # Similarity score
        similarity_scores.append(string_similarity(pred, gt))

    total = len(test_data)
    exact_acc = exact_matches / total if total else 0
    avg_sim = sum(similarity_scores) / total if total else 0
    return {"exact_accuracy": exact_acc, "avg_similarity": avg_sim}

def main():
    results_table = []

    for field in FIELDS:
        row = [field.upper()]
        for engine, ocr_dir in OCR_DIRS.items():
            model_path = ENSEMBLE_MODEL_DIRS[engine] / f"{field}_ensemble.json"
            hybrid_extractor = HybridExtractor(field, ensemble_model_path=model_path)
            test_data = load_test_data(ocr_dir, field)
            metrics = evaluate_hybrid(hybrid_extractor, test_data)
            row.append(f"{metrics['exact_accuracy']:.4f}")
        # Determine winner
        t_acc = float(row[1])
        p_acc = float(row[2])
        winner = "Tesseract" if t_acc > p_acc else "PaddleOCR"
        row.append(winner)
        results_table.append(row)

    headers = ["Field", "Tesseract", "PaddleOCR", "Winner"]
    print("\n" + "="*80)
    print("HYBRID EXTRACTOR PERFORMANCE ON TEST SET")
    print("="*80)
    print(tabulate(results_table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
