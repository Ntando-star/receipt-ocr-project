# scripts/standardize_test_results.py
import json
from pathlib import Path
from difflib import SequenceMatcher
from src.ensemble_regex_extractor import EnsembleRegexExtractor

# -----------------------------
# Configuration
# -----------------------------
FIELDS = ["company", "address", "date", "total"]

OCR_ENGINES = {
    "tesseract": "data/ocr_output_tesseract_test",
    "paddle": "data/ocr_output_paddle_test"
}

MODEL_DIRS = {
    "tesseract": Path("data/models/ensemble_tesseract"),
    "paddle": Path("data/models/ensemble_paddle")
}

GROUND_TRUTH_DIR = Path("data/raw/ground_truth")
OUTPUT_DIR = Path("data/standardized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILES = {
    "tesseract": OUTPUT_DIR / "tesseract_baseline.json",
    "paddle": OUTPUT_DIR / "paddle_ensemble.json"
}

# -----------------------------
# Helper functions
# -----------------------------
def string_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 1.0 if (not s1 and not s2) else 0.0
    return SequenceMatcher(None, s1.lower().strip(), s2.lower().strip()).ratio()


def load_ground_truth(filename: str) -> dict:
    path = GROUND_TRUTH_DIR / filename
    if path.exists():
        try:
            data = json.load(open(path, encoding="utf-8"))
            return {field: data.get(field, "") for field in FIELDS}
        except json.JSONDecodeError:
            pass
    return {field: "" for field in FIELDS}


def load_ensemble_models(engine: str) -> dict:
    models = {}
    for field in FIELDS:
        model_path = MODEL_DIRS[engine] / f"{field}_ensemble.json"
        extractor = EnsembleRegexExtractor(field_name=field)
        if model_path.exists():
            extractor.load_model(str(model_path))
        models[field] = extractor
    return models


def hybrid_extract(ocr_text: str, extractor: EnsembleRegexExtractor, field: str) -> str:
    """Use ensemble regex, fallback to simple heuristics"""
    result = extractor.extract_with_confidence(ocr_text).get("value", "")
    if not result:
        lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
        if field == "company":
            for line in lines:
                if any(k in line.upper() for k in ["SDN BHD", "LTD", "ENTERPRISE", "PLT"]):
                    result = line
                    break
        elif field == "address":
            for line in lines:
                if any(k in line.lower() for k in ["jalan", "st", "street", "ave", "no."]):
                    result = line
                    break
    return result


# -----------------------------
# Main processing
# -----------------------------
for engine, ocr_dir in OCR_ENGINES.items():
    ocr_dir = Path(ocr_dir)
    extractors = load_ensemble_models(engine)

    results = {
        "test_results": {
            "baseline": {field: [] for field in FIELDS},
            "ensemble": {field: [] for field in FIELDS}
        }
    }

    for gt_file in sorted(GROUND_TRUTH_DIR.glob("*.txt")):
        gt_data = load_ground_truth(gt_file.name)
        ocr_file = ocr_dir / gt_file.name
        ocr_text = ocr_file.read_text(encoding="utf-8") if ocr_file.exists() else ""

        for field in FIELDS:
            pred = hybrid_extract(ocr_text, extractors[field], field)
            sim = string_similarity(pred, gt_data.get(field, ""))
            results["test_results"]["ensemble"][field].append({
                "filename": gt_file.name,
                "prediction": pred,
                "ground_truth": gt_data.get(field, ""),
                "similarity": sim
            })
            # Optionally, also copy to baseline if you want baseline = ensemble for now
            results["test_results"]["baseline"][field].append({
                "filename": gt_file.name,
                "prediction": pred,
                "ground_truth": gt_data.get(field, ""),
                "similarity": sim
            })

    # Save standardized JSON
    output_path = OUTPUT_FILES[engine]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Standardized results saved for {engine} at {output_path}")
