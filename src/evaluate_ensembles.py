# src/evaluate_ensembles.py
import json
from pathlib import Path
from src.ensemble_regex_extractor import EnsembleRegexExtractor
from tabulate import tabulate  # pip install tabulate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Config
# -----------------------
FIELDS = ["company", "address", "date", "total"]
OCR_ENGINES = ["tesseract", "paddle"]

GT_DIR = Path("data/raw/ground_truth")
MODEL_DIRS = {
    "tesseract": Path("data/models/ensemble_tesseract"),
    "paddle": Path("data/models/ensemble_paddle")
}
TEST_DIRS = {
    "tesseract": Path("data/ocr_output_tesseract_test"),
    "paddle": Path("data/ocr_output_paddle_test")
}

# -----------------------
# Helper functions
# -----------------------
def load_test_data(ocr_dir: Path, field: str):
    """Load OCR text and ground truth for the test set."""
    data = []
    for gt_file in GT_DIR.glob("*.txt"):
        with open(gt_file, "r") as f:
            gt = json.load(f)
        ocr_file = ocr_dir / gt_file.name
        ocr_text = ocr_file.read_text(encoding="utf-8") if ocr_file.exists() else ""
        data.append({"ocr_text": ocr_text, "ground_truth": gt.get(field, "")})
    return data

def evaluate_model(extractor: EnsembleRegexExtractor, test_data: list):
    """Evaluate exact match and average similarity on test data."""
    exact_matches = 0
    similarity_scores = []

    for sample in test_data:
        ocr_text = sample["ocr_text"]
        gt = sample["ground_truth"]

        # Preprocess OCR text for better regex matching
        ocr_text_processed = extractor.preprocess_ocr_text(ocr_text)
        result = extractor.extract_with_confidence(ocr_text_processed)
        pred = result["value"] or ""

        # Exact match
        if pred.strip() == gt.strip():
            exact_matches += 1

        # Similarity score
        similarity_scores.append(extractor._string_similarity(pred, gt))

    total = len(test_data)
    exact_acc = exact_matches / total if total else 0
    avg_sim = sum(similarity_scores) / total if total else 0
    return {"exact_accuracy": exact_acc, "avg_similarity": avg_sim}

# -----------------------
# Main evaluation loop
# -----------------------
def main():
    results_table = []

    for field in FIELDS:
        row = [field.upper()]
        for engine in OCR_ENGINES:
            # Load model
            model_path = MODEL_DIRS[engine] / f"{field}_ensemble.json"
            extractor = EnsembleRegexExtractor(field)
            success = extractor.load_model(model_path)
            if not success:
                logger.warning(f"Could not load model for {engine} {field}")
                row.append("N/A")
                continue

            # Load test data
            test_data = load_test_data(TEST_DIRS[engine], field)

            # Evaluate
            metrics = evaluate_model(extractor, test_data)
            row.append(f"{metrics['exact_accuracy']:.4f}")
        # Determine winner
        try:
            t_acc = float(row[1])
            p_acc = float(row[2])
            winner = "Tesseract" if t_acc > p_acc else "PaddleOCR"
        except ValueError:
            winner = "N/A"
        row.append(winner)
        results_table.append(row)

    headers = ["Field", "Tesseract", "PaddleOCR", "Winner"]
    print("\n" + "="*80)
    print("ENSEMBLE REGEX PERFORMANCE ON TEST SET")
    print("="*80)
    print(tabulate(results_table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
