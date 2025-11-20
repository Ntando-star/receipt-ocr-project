import json
from pathlib import Path
import logging
from difflib import SequenceMatcher
from src.ensemble_regex_extractor import EnsembleRegexExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OCR_ENGINES = ["tesseract", "paddle"]
FIELDS = ["company", "address", "date", "total"]

GT_DIR = Path("data/raw/ground_truth")

OCR_SPLITS = {
    "train": {
        "tesseract": Path("data/ocr_output_tesseract"),
        "paddle": Path("data/ocr_output_paddle")
    },
    "val": {
        "tesseract": Path("data/ocr_output_tesseract_val"),
        "paddle": Path("data/ocr_output_paddle_val")
    },
    "test": {
        "tesseract": Path("data/ocr_output_tesseract_test"),
        "paddle": Path("data/ocr_output_paddle_test")
    },
}

MODEL_OUTPUT_DIRS = {
    "tesseract": Path("data/models/ensemble_tesseract"),
    "paddle": Path("data/models/ensemble_paddle")
}

for mp in MODEL_OUTPUT_DIRS.values():
    mp.mkdir(parents=True, exist_ok=True)


def string_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 1.0 if not s1 and not s2 else 0.0
    return SequenceMatcher(None, s1.lower().strip(), s2.lower().strip()).ratio()


def load_split_data(split: str, engine: str, field: str):
    data = []
    ocr_dir = OCR_SPLITS[split][engine]

    for ocr_file in sorted(ocr_dir.glob("*.txt")):
        gt_file = GT_DIR / ocr_file.name
        if not gt_file.exists():
            continue

        with open(gt_file, "r", encoding="utf-8") as f:
            gt = json.load(f)

        ocr_text = ocr_file.read_text(encoding="utf-8")
        data.append({"ocr_text": ocr_text, "ground_truth": gt.get(field, "")})

    return data


def compute_metrics(preds, gts):
    return sum(string_similarity(p, g) for p, g in zip(preds, gts)) / len(preds)


def train_and_eval_field(field: str):
    logger.info(f"\n=== Training '{field}' ===")
    extractors = {}

    for engine in OCR_ENGINES:
        train_data = load_split_data("train", engine, field)
        val_data = load_split_data("val", engine, field)
        test_data = load_split_data("test", engine, field)

        if not train_data:
            logger.warning(f"No training data for {engine} / {field}")
            continue

        extractor = EnsembleRegexExtractor(field)
        train_metrics = extractor.learn_ensemble(train_data)
        extractor.fine_tune(val_data)

        model_path = MODEL_OUTPUT_DIRS[engine] / f"{field}_ensemble.json"
        extractor.save_model(str(model_path))
        logger.info(f"📦 Saved: {model_path}")

        logger.info(f"🏋️ Train avg sim: {train_metrics['average_accuracy']:.4f}")

        test_metrics = extractor.evaluate(test_data)
        logger.info(f"🧪 Test avg sim ({engine}): {test_metrics['avg_similarity']:.4f}")

        extractors[engine] = extractor

    return extractors


def main():
    for field in FIELDS:
        train_and_eval_field(field)


if __name__ == "__main__":
    main()
