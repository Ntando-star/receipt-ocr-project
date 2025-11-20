# run_complete_pipeline.py
"""
Complete OCR Research Pipeline
Uses train_ensemble_pipeline.main() directly for ensemble training + evaluation
"""

import logging
from src import train_ensemble_pipeline
from src.comparison_analyzer import ComparisonAnalyzer, generate_comparison_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Starting Ensemble Training + Evaluation ===")
    # Run the training & evaluation pipeline for all fields
    train_ensemble_pipeline.main()

    # Standardized JSON results (make sure these exist after the above pipeline)
    tesseract_results = 'data/standardized/tesseract_baseline.json'
    paddle_results = 'data/standardized/paddle_ensemble.json'

    logger.info("=== Running Comparative Analysis ===")
    analyzer = ComparisonAnalyzer(tesseract_results, paddle_results)
    comparison = analyzer.analyze_all_fields()
    analyzer.save_analysis('results/comprehensive_comparison.json')

    logger.info("=== Generating Markdown Report ===")
    generate_comparison_report('results/comprehensive_comparison.json', 'FINAL_RESULTS.md')

    logger.info("✅ Pipeline complete")

if __name__ == "__main__":
    main()
