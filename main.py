# main.py
import os
import sys
import logging
from pathlib import Path
from paddleocr import PaddleOCR

from src.config import config
from src.ensemble_evaluation import EnsembleEvaluator
from src.extract_fields import extract_fields as extract_tesseract_baseline
from src.extract_fields_paddle import extract_fields as extract_paddle_baseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
TRAIN_TESSERACT_DIR = "data/ocr_output_tesseract"  # Tesseract training output
TRAIN_PADDLE_DIR = "data/ocr_output_paddle"  # Paddle training output
TEST_RAW_DIR = "data/raw/test_data"  # Raw test images
TEST_TESSERACT_DIR = "data/ocr_output_paddle_test"
TEST_PADDLE_DIR = "data/ocr_output_paddle_test"
VAL_TESSERACT_DIR = "data/ocr_output_tesseract_val"
VAL_PADDLE_DIR = "data/ocr_output_paddle_val"
GT_DIR = "data/ground_truth"

# Create output dirs
os.makedirs(TEST_TESSERACT_DIR, exist_ok=True)
os.makedirs(TEST_PADDLE_DIR, exist_ok=True)

# Initialize PaddleOCR
ocr_paddle = PaddleOCR(use_textline_orientation=True, lang='en')


def run_paddleocr_on_file(image_path):
    try:
        result = ocr_paddle.ocr(image_path)
        if result:
            texts = [line[1][0] for line in result[0]]
            return "\n".join(texts)
        return ""
    except Exception as e:
        logger.error(f"PaddleOCR error for {image_path}: {e}")
        return ""


def run_tesseract_on_file(image_path):
    """Assumes tesseract is installed and pytesseract is available"""
    import pytesseract
    from PIL import Image
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        logger.error(f"Tesseract error for {image_path}: {e}")
        return ""


def process_ocr_on_test(raw_dir, output_dir, engine="tesseract"):
    """Run OCR on test images"""
    image_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    logger.info(f"Found {len(image_files)} test images for {engine}")
    for fname in image_files:
        image_path = os.path.join(raw_dir, fname)
        if engine == "tesseract":
            ocr_text = run_tesseract_on_file(image_path)
        else:
            ocr_text = run_paddleocr_on_file(image_path)

        if ocr_text:
            out_path = os.path.join(output_dir, Path(fname).stem + ".txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            logger.info(f"✅ {engine} OCR saved: {out_path}")
        else:
            logger.warning(f"No OCR output for {fname}")


def compute_baseline_results(ocr_dir, gt_dir, engine="tesseract"):
    """Run standard regex baseline"""
    evaluator = EnsembleEvaluator(ocr_dir, gt_dir)
    if engine == "tesseract":
        baseline_func = extract_tesseract_baseline
    else:
        baseline_func = extract_paddle_baseline

    field_results = {}
    for fname in sorted(os.listdir(ocr_dir)):
        if not fname.endswith(".txt"):
            continue
        ocr_path = os.path.join(ocr_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            continue

        with open(ocr_path, 'r', encoding='utf-8') as f:
            ocr_text = f.read()
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        pred_data = baseline_func(ocr_text)
        for field in evaluator.FIELDS:
            pred = pred_data.get(field)
            gt = gt_data.get(field)
            similarity = evaluator._string_similarity(pred, gt)
            field_results.setdefault(field, []).append({
                "predicted": pred,
                "ground_truth": gt,
                "similarity": similarity,
                "filename": fname
            })

    metrics = evaluator._compute_metrics(field_results)
    return {"field_results": field_results, "metrics": metrics}


def run_pipeline():
    # ---------------------------
    # 1️⃣ Run OCR on test images
    # ---------------------------
    logger.info("\n===== OCR ON TEST DATA =====")
    process_ocr_on_test(TEST_RAW_DIR, TEST_TESSERACT_DIR, engine="tesseract")
    process_ocr_on_test(TEST_RAW_DIR, TEST_PADDLE_DIR, engine="paddle")

    # ---------------------------
    # 2️⃣ Baseline Extraction
    # ---------------------------
    logger.info("\n===== BASELINE EXTRACTION =====")
    tesseract_baseline = compute_baseline_results(TEST_TESSERACT_DIR, GT_DIR, engine="tesseract")
    paddle_baseline = compute_baseline_results(TEST_PADDLE_DIR, GT_DIR, engine="paddle")

    # Save baseline results
    with open("results_tesseract_baseline.json", "w") as f:
        json.dump(tesseract_baseline, f, indent=2)
    with open("results_paddle_baseline.json", "w") as f:
        json.dump(paddle_baseline, f, indent=2)
    logger.info("✅ Baseline results saved")

    # ---------------------------
    # 3️⃣ Train Ensemble Models
    # ---------------------------
    logger.info("\n===== TRAINING ENSEMBLE MODELS =====")
    evaluator_tess = EnsembleEvaluator(TRAIN_TESSERACT_DIR, GT_DIR)
    evaluator_paddle = EnsembleEvaluator(TRAIN_PADDLE_DIR, GT_DIR)

    train_data_tess = evaluator_tess.load_training_data()
    train_data_paddle = evaluator_paddle.load_training_data()

    models_tess = evaluator_tess.train_ensemble_models(train_data_tess)
    models_paddle = evaluator_paddle.train_ensemble_models(train_data_paddle)

    # ---------------------------
    # 4️⃣ Ensemble Evaluation on Test Data
    # ---------------------------
    logger.info("\n===== ENSEMBLE EVALUATION =====")
    results_tess_ensemble = evaluator_tess.evaluate_models(models_tess, test_ocr_dir=TEST_TESSERACT_DIR)
    results_paddle_ensemble = evaluator_paddle.evaluate_models(models_paddle, test_ocr_dir=TEST_PADDLE_DIR)

    # ---------------------------
    # 5️⃣ Save Ensemble Results
    # ---------------------------
    evaluator_tess.save_results(results_tess_ensemble, output_path="results_tesseract_ensemble.json")
    evaluator_paddle.save_results(results_paddle_ensemble, output_path="results_paddle_ensemble.json")
    logger.info("✅ Ensemble results saved")

    # ---------------------------
    # 6️⃣ Print summary
    # ---------------------------
    logger.info("\n===== SUMMARY: TESSERACT =====")
    evaluator_tess.print_summary(results_tess_ensemble)
    logger.info("\n===== SUMMARY: PADDLEOCR =====")
    evaluator_paddle.print_summary(results_paddle_ensemble)


if __name__ == "__main__":
    run_pipeline()
