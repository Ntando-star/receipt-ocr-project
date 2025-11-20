#!/usr/bin/env python3  
# run_quick_evaluation.py  
"""  
Quick evaluation without full training (for testing/debugging)  
"""  
  
import sys  
from pathlib import Path  
sys.path.insert(0, str(Path(__file__).resolve().parent))  
  
import logging  
from src.config import config  
from src.data_loader import DataLoader  
from src.extract_fields import extract_fields as extract_tesseract  
from src.extract_fields_paddle import extract_fields as extract_paddle  
from difflib import SequenceMatcher  
import numpy as np  
  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
def string_similarity(s1, s2):  
    if not s1 or not s2:  
        return 1.0 if (not s1 and not s2) else 0.0  
    s1_clean = str(s1).lower().strip()  
    s2_clean = str(s2).lower().strip()  
    return SequenceMatcher(None, s1_clean, s2_clean).ratio()  
  
def quick_eval():  
    """Quick baseline evaluation"""  
      
    logger.info("\n" + "="*70)  
    logger.info("QUICK BASELINE EVALUATION")  
    logger.info("="*70)  
      
    loader = DataLoader(config)  
      
    # Load test data  
    _, _, test_tesseract = loader.load_tesseract_data()  
    _, _, test_paddle = loader.load_paddle_data()  
      
    logger.info(f"\nTesseract test samples: {len(test_tesseract)}")  
    logger.info(f"PaddleOCR test samples: {len(test_paddle)}")  
      
    # Evaluate Tesseract  
    logger.info("\n" + "-"*70)  
    logger.info("TESSERACT BASELINE")  
    logger.info("-"*70)  
      
    tess_results = {field: [] for field in config.FIELDS}  
    for sample in test_tesseract:  
        extracted = extract_tesseract(sample.ocr_text)  
        for field in config.FIELDS:  
            pred = extracted.get(field) or ""  
            gt = sample.ground_truth.get(field) or ""  
            sim = string_similarity(pred, gt)  
            tess_results[field].append(sim)  
      
    for field in config.FIELDS:  
        sims = tess_results[field]  
        if sims:  
            correct = sum(1 for s in sims if s >= config.SIMILARITY_THRESHOLD)  
            pct = 100 * correct / len(sims)  
            logger.info(f"  {field:<12}: {np.mean(sims):.4f} ({correct}/{len(sims)} = {pct:.1f}%)")  
      
    # Evaluate PaddleOCR  
    logger.info("\n" + "-"*70)  
    logger.info("PADDLEOCR BASELINE")  
    logger.info("-"*70)  
      
    paddle_results = {field: [] for field in config.FIELDS}  
    for sample in test_paddle:  
        extracted = extract_paddle(sample.ocr_text)  
        for field in config.FIELDS:  
            pred = extracted.get(field) or ""  
            gt = sample.ground_truth.get(field) or ""  
            sim = string_similarity(pred, gt)  
            paddle_results[field].append(sim)  
      
    for field in config.FIELDS:  
        sims = paddle_results[field]  
        if sims:  
            correct = sum(1 for s in sims if s >= config.SIMILARITY_THRESHOLD)  
            pct = 100 * correct / len(sims)  
            logger.info(f"  {field:<12}: {np.mean(sims):.4f} ({correct}/{len(sims)} = {pct:.1f}%)")  
      
    # Comparison  
    logger.info("\n" + "-"*70)  
    logger.info("COMPARISON")  
    logger.info("-"*70)  
      
    for field in config.FIELDS:  
        tess_mean = np.mean(tess_results[field]) if tess_results[field] else 0  
        paddle_mean = np.mean(paddle_results[field]) if paddle_results[field] else 0  
        diff = paddle_mean - tess_mean  
        logger.info(f"  {field:<12}: Tess={tess_mean:.4f}, Paddle={paddle_mean:.4f}, Diff={diff:+.4f}")  
  
if __name__ == "__main__":  
    quick_eval()  