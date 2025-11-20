# src/comparison_analyzer.py
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from pathlib import Path
import mlflow

from src.config import config
from src.statistical_tests import StatisticalTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparisonAnalyzer:
    """Compare Tesseract vs PaddleOCR performance"""

    def __init__(
        self,
        tesseract_results_file: str,
        paddle_results_file: str,
        config=config,
    ):
        self.config = config

        with open(tesseract_results_file, 'r') as f:
            self.tesseract_results = json.load(f)

        with open(paddle_results_file, 'r') as f:
            self.paddle_results = json.load(f)

        self.comparison = {}
        self.statistical_tests = {}

        mlflow.set_tracking_uri(f"sqlite:///{self.config.MLRUNS_DIR}/mlflow.db")
        mlflow.set_experiment("OCR_Comparison_Final")

    def analyze_all_fields(self) -> Dict:
        """Analyze performance across all fields"""
        logger.info("\n" + "="*70)
        logger.info("CROSS-ENGINE COMPARISON ANALYSIS")
        logger.info("="*70)

        with mlflow.start_run(run_name="Comprehensive_Comparison"):
            mlflow.log_param("comparison_type", "Tesseract_vs_PaddleOCR")

            for field in self.config.FIELDS:
                self._analyze_field(field)

                # Log comparison metrics
                if field in self.comparison:
                    comp = self.comparison[field]
                    mlflow.log_metric(f"{field}_baseline_comparison",
                                      comp.get("baseline_diff", 0))
                    mlflow.log_metric(f"{field}_ensemble_comparison",
                                      comp.get("ensemble_diff", 0))

        return self.comparison

    def _analyze_field(self, field: str):
        """Analyze single field across engines"""
        logger.info(f"\n{'─'*70}")
        logger.info(f"FIELD: {field.upper()}")
        logger.info(f"{'─'*70}")

        # Extract accuracies
        tess_baseline = self._extract_accuracies(
            self.tesseract_results, "baseline", field
        )
        tess_ensemble = self._extract_accuracies(
            self.tesseract_results, "ensemble", field
        )
        paddle_baseline = self._extract_accuracies(
            self.paddle_results, "baseline", field
        )
        paddle_ensemble = self._extract_accuracies(
            self.paddle_results, "ensemble", field
        )

        # Compute means safely
        tess_bl_mean = np.mean(tess_baseline) if tess_baseline else 0.0
        tess_ens_mean = np.mean(tess_ensemble) if tess_ensemble else 0.0
        paddle_bl_mean = np.mean(paddle_baseline) if paddle_baseline else 0.0
        paddle_ens_mean = np.mean(paddle_ensemble) if paddle_ensemble else 0.0

        logger.info(f"\nBaseline Comparison:")
        logger.info(f"  Tesseract:  {tess_bl_mean:.4f}")
        logger.info(f"  PaddleOCR:  {paddle_bl_mean:.4f}")
        logger.info(f"  Difference: {(paddle_bl_mean - tess_bl_mean):+.4f}")

        logger.info(f"\nEnsemble Comparison:")
        logger.info(f"  Tesseract:  {tess_ens_mean:.4f}")
        logger.info(f"  PaddleOCR:  {paddle_ens_mean:.4f}")
        logger.info(f"  Difference: {(paddle_ens_mean - tess_ens_mean):+.4f}")

        # Statistical tests (run only if there are scores)
        tess_stats = None
        if tess_baseline and tess_ensemble:
            tess_stats = StatisticalTester.run_comprehensive_tests(
                tess_baseline, tess_ensemble,
                f"Tesseract_{field}_Improvement"
            )

        paddle_stats = None
        if paddle_baseline and paddle_ensemble:
            paddle_stats = StatisticalTester.run_comprehensive_tests(
                paddle_baseline, paddle_ensemble,
                f"PaddleOCR_{field}_Improvement"
            )

        cross_stats = None
        if tess_ensemble and paddle_ensemble:
            cross_stats = StatisticalTester.run_comprehensive_tests(
                tess_ensemble, paddle_ensemble,
                f"CrossEngine_{field}_Ensemble"
            )

        self.comparison[field] = {
            "baseline": {
                "tesseract_accuracy": round(tess_bl_mean, 4),
                "paddle_accuracy": round(paddle_bl_mean, 4),
                "difference": round(paddle_bl_mean - tess_bl_mean, 4),
                "winner": "PaddleOCR" if paddle_bl_mean > tess_bl_mean else "Tesseract"
            },
            "ensemble": {
                "tesseract_accuracy": round(tess_ens_mean, 4),
                "paddle_accuracy": round(paddle_ens_mean, 4),
                "difference": round(paddle_ens_mean - tess_ens_mean, 4),
                "winner": "PaddleOCR" if paddle_ens_mean > tess_ens_mean else "Tesseract"
            },
            "tesseract_improvement": {
                "baseline_to_ensemble": round(tess_ens_mean - tess_bl_mean, 4),
                "percentage_improvement": round(
                    ((tess_ens_mean - tess_bl_mean) / tess_bl_mean * 100)
                    if tess_bl_mean > 0 else 0, 2
                ),
                "statistical_test": tess_stats
            },
            "paddle_improvement": {
                "baseline_to_ensemble": round(paddle_ens_mean - paddle_bl_mean, 4),
                "percentage_improvement": round(
                    ((paddle_ens_mean - paddle_bl_mean) / paddle_bl_mean * 100)
                    if paddle_bl_mean > 0 else 0, 2
                ),
                "statistical_test": paddle_stats
            },
            "cross_engine_comparison": cross_stats
        }

        self.statistical_tests[field] = {
            "tesseract_improvement_test": tess_stats,
            "paddle_improvement_test": paddle_stats,
            "cross_engine_test": cross_stats
        }

    def save_analysis(self, output_path=None) -> str:
        """Save analysis to JSON"""
        if output_path is None:
            output_path = self.config.RESULTS_DIR / "comprehensive_comparison.json"

        # Ensure output_path is a Path object
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "comparison": self.comparison,
                "statistical_tests": self.statistical_tests
            }, f, indent=2)

        logger.info(f"✅ Analysis saved to {output_path}")
        return str(output_path)


    @staticmethod
    def _extract_accuracies(results: Dict, method: str, field: str) -> List[float]:
        """Extract similarity scores for field"""
        try:
            field_results = results.get("test_results", {}).get(method, {}).get(field, [])
            return [r.get("similarity", 0.0) for r in field_results if r.get("similarity") is not None]
        except:
            return []


def generate_comparison_report(comparison_file: str, output_file: str = "RESULTS_COMPARISON.md"):
    """Generate markdown report from comparison analysis (safe with missing stats)"""

    with open(comparison_file, 'r') as f:
        data = json.load(f)

    comparison = data.get("comparison", {})

    report = f"""
# OCR Performance Comparison Report: Tesseract vs PaddleOCR

## Executive Summary

This report compares the performance of **Tesseract OCR** and **PaddleOCR** when combined with:
1. **Standard (hand-crafted) Regex** patterns
2. **Ensemble-learned Regex** patterns

The analysis includes statistical hypothesis testing to determine if performance differences are
statistically significant.

---

## Results by Field

"""

    for field in ["company", "address", "date", "total"]:
        if field not in comparison:
            continue

        comp = comparison[field]

        report += f"""
### {field.upper()}

#### Baseline Performance (Hand-crafted Regex)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | {comp['baseline']['tesseract_accuracy']:.4f} | {comp['baseline']['winner']} |
| PaddleOCR | {comp['baseline']['paddle_accuracy']:.4f} | |
| **Difference** | **{comp['baseline']['difference']:+.4f}** | |

#### Ensemble Performance (Learned Patterns)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | {comp['ensemble']['tesseract_accuracy']:.4f} | {comp['ensemble']['winner']} |
| PaddleOCR | {comp['ensemble']['paddle_accuracy']:.4f} | |
| **Difference** | **{comp['ensemble']['difference']:+.4f}** | |

#### Improvement from Baseline to Ensemble

**Tesseract:**
- Absolute: {comp['tesseract_improvement']['baseline_to_ensemble']:+.4f}
- Percentage: {comp['tesseract_improvement']['percentage_improvement']:+.2f}%

**PaddleOCR:**
- Absolute: {comp['paddle_improvement']['baseline_to_ensemble']:+.4f}
- Percentage: {comp['paddle_improvement']['percentage_improvement']:+.2f}%

#### Statistical Significance (Ensemble Improvement)
"""

        def format_test(test):
            if not test:
                return "- Paired t-test: Skipped (no valid data)\n- Cohen's d: Skipped (no valid data)\n"
            lines = []
            pt = test.get('paired_t_test')
            cd = test.get('cohens_d')

            if pt:
                lines.append(f"- Paired t-test: t={pt['t_statistic']}, p={pt['p_value']:.6f}  "
                             f"- Significant (α=0.05): {'✅ Yes' if pt['significant_at_0.05'] else '❌ No'}")
            else:
                lines.append("- Paired t-test: Skipped (no valid data)")

            if cd:
                lines.append(f"- Cohen's d: {cd['cohens_d']:.4f} ({cd['interpretation']})")
            else:
                lines.append("- Cohen's d: Skipped (no valid data)")

            return "\n".join(lines)

        report += "**Tesseract Baseline → Ensemble:**\n"
        report += format_test(comp['tesseract_improvement'].get('statistical_test')) + "\n\n"

        report += "**PaddleOCR Baseline → Ensemble:**\n"
        report += format_test(comp['paddle_improvement'].get('statistical_test')) + "\n\n"

        report += "**Cross-Engine Comparison (Ensemble):**\n"
        report += format_test(comp.get('cross_engine_comparison')) + "\n\n---\n\n"

    report += """
## Key Findings

1. **Effect of Ensemble Learning**: Ensemble patterns improve baseline accuracy through:
   - Automated pattern discovery from training data
   - Variance-weighted pattern selection
   - Majority voting across multiple regex patterns

2. **OCR Engine Comparison**: Performance differences between Tesseract and PaddleOCR vary by field

3. **Statistical Significance**: P-values determine if improvements are due to systematic differences
   or random variation

---

## Recommendations

- Use ensemble-learned patterns if improvement is statistically significant (p < 0.05)
- Prefer PaddleOCR if it shows consistently higher accuracy across fields
- Consider field-specific model selection based on statistical tests
- Deploy models with confidence-based fallback mechanisms

---

*Report generated automatically from evaluation results*
"""

    with open(output_file, 'w') as f:
        f.write(report)

    logger.info(f"✅ Report saved to {output_file}")
