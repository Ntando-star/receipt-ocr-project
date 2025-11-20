# evaluate_ocr_performance.py
import os
import json
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
import sys

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.extract_fields import extract_fields as extract_tesseract
from src.extract_fields_paddle import extract_fields as extract_paddle

# ================================================================================  
# CONFIGURATION  
# ================================================================================  

TESSERACT_OCR_DIR = "data/ocr_output_tesseract_test"
PADDLE_OCR_DIR = "data/ocr_output_paddle_test"
GROUND_TRUTH_DIR = "data/raw/ground_truth"

FIELDS = ["company", "address", "date", "total"]

# ================================================================================  
# HELPER FUNCTIONS  
# ================================================================================  

def string_similarity(s1: str, s2: str) -> float:
    """Calculate string similarity (0-1) using SequenceMatcher."""
    if not s1 or not s2:
        return 1.0 if (not s1 and not s2) else 0.0
    s1_clean = str(s1).lower().strip()
    s2_clean = str(s2).lower().strip()
    return SequenceMatcher(None, s1_clean, s2_clean).ratio()


def load_ocr_text(ocr_dir: str, filename: str) -> str:
    """Load OCR text from file."""
    ocr_path = os.path.join(ocr_dir, filename)
    if os.path.exists(ocr_path):
        with open(ocr_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def load_ground_truth(gt_dir: str, filename: str) -> Dict:
    """Load ground truth JSON."""
    gt_path = os.path.join(gt_dir, filename)
    if os.path.exists(gt_path):
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {field: data.get(field) for field in FIELDS}
        except json.JSONDecodeError:
            pass
    return {field: None for field in FIELDS}


# ================================================================================  
# STATISTICAL TESTS  
# ================================================================================  

def paired_t_test(tesseract_scores: List[float], paddle_scores: List[float]) -> Dict:
    min_len = min(len(tesseract_scores), len(paddle_scores))
    t_stat, p_value = stats.ttest_rel(tesseract_scores[:min_len], paddle_scores[:min_len])
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_0.05": bool(p_value < 0.05),
        "significant_0.01": bool(p_value < 0.01)
    }


def mann_whitney_u(tesseract_scores: List[float], paddle_scores: List[float]) -> Dict:
    min_len = min(len(tesseract_scores), len(paddle_scores))
    u_stat, p_value = stats.mannwhitneyu(tesseract_scores[:min_len], paddle_scores[:min_len])
    return {
        "u_statistic": round(float(u_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_0.05": bool(p_value < 0.05)
    }


def cohens_d(x: List[float], y: List[float]) -> float:
    min_len = min(len(x), len(y))
    x_arr = np.array(x[:min_len])
    y_arr = np.array(y[:min_len])
    pooled_std = np.sqrt((np.std(x_arr, ddof=1) ** 2 + np.std(y_arr, ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x_arr) - np.mean(y_arr)) / pooled_std)


# ================================================================================  
# MAIN EVALUATION CLASS  
# ================================================================================  

class OCRPerformanceEvaluator:
    """Comprehensive OCR performance evaluation with statistical analysis."""

    def __init__(self, tesseract_dir: str, paddle_dir: str, gt_dir: str):
        self.tesseract_dir = tesseract_dir
        self.paddle_dir = paddle_dir
        self.gt_dir = gt_dir
        self.results = {"tesseract": {}, "paddle": {}}

    def evaluate_all(self) -> Dict:
        """Run complete evaluation pipeline."""
        # Get all files
        ocr_files = sorted([f for f in os.listdir(self.tesseract_dir) if f.endswith(".txt")])

        # Evaluate engines
        self.results["tesseract"] = self._evaluate_engine(self.tesseract_dir, extract_tesseract, ocr_files)
        self.results["paddle"] = self._evaluate_engine(self.paddle_dir, extract_paddle, ocr_files)

        # Compute comparison statistics
        self.results["comparison"] = self._compute_comparison_statistics()

        return self.results

    def _evaluate_engine(self, ocr_dir: str, extractor_func, ocr_files: List[str]) -> Dict:
        """Evaluate a single OCR engine."""
        field_results = {field: [] for field in FIELDS}
        file_count = 0

        for filename in ocr_files:
            ocr_text = load_ocr_text(ocr_dir, filename)
            if not ocr_text:
                continue

            gt_data = load_ground_truth(self.gt_dir, filename)
            if not any(gt_data.values()):
                continue

            try:
                extracted = extractor_func(ocr_text)
            except Exception:
                extracted = {field: "" for field in FIELDS}

            for field in FIELDS:
                pred = extracted.get(field) or ""
                gt = gt_data.get(field) or ""
                similarity = string_similarity(pred, gt)

                field_results[field].append({
                    "similarity": similarity,
                    "correct": similarity >= 0.8
                })

            file_count += 1

        stats = self._compute_field_statistics(field_results)
        return {"field_results": field_results, "statistics": stats, "files_processed": file_count}

    def _compute_field_statistics(self, field_results: Dict[str, List[Dict]]) -> Dict:
        stats_dict = {}
        for field, results in field_results.items():
            if not results:
                stats_dict[field] = self._empty_stats()
                continue
            similarities = [r["similarity"] for r in results]
            correct_count = sum(1 for r in results if r["correct"])
            stats_dict[field] = {
                "accuracy": round(np.mean(similarities), 4),
                "accuracy_std": round(np.std(similarities), 4),
                "accuracy_min": round(np.min(similarities), 4),
                "accuracy_max": round(np.max(similarities), 4),
                "accuracy_median": round(np.median(similarities), 4),
                "correct_count": correct_count,
                "correct_percentage": round(correct_count / len(results) * 100, 2),
                "incorrect_count": len(results) - correct_count,
                "n_samples": len(results)
            }
        return stats_dict

    def _empty_stats(self) -> Dict:
        return {key: 0.0 for key in ["accuracy", "accuracy_std", "accuracy_min",
                                     "accuracy_max", "accuracy_median",
                                     "correct_count", "correct_percentage",
                                     "incorrect_count", "n_samples"]}

    def _compute_comparison_statistics(self) -> Dict:
        comparison = {}
        for field in FIELDS:
            tess = self.results["tesseract"]["statistics"].get(field, self._empty_stats())
            paddle = self.results["paddle"]["statistics"].get(field, self._empty_stats())

            tess_acc = tess["accuracy"]
            paddle_acc = paddle["accuracy"]
            abs_diff = paddle_acc - tess_acc
            pct_improvement = (abs_diff / tess_acc * 100) if tess_acc > 0 else 0

            if paddle_acc > tess_acc:
                winner = "PaddleOCR"
            elif tess_acc > paddle_acc:
                winner = "Tesseract"
            else:
                winner = "Tie"

            comparison[field] = {
                "tesseract_accuracy": round(tess_acc, 4),
                "paddle_accuracy": round(paddle_acc, 4),
                "absolute_difference": round(abs_diff, 4),
                "percentage_improvement": round(pct_improvement, 2),
                "winner": winner
            }
        return comparison

    def print_summary(self):
        """Print overall evaluation summary."""
        for engine in ["tesseract", "paddle"]:
            stats = self.results[engine]["statistics"]
            files = self.results[engine]["files_processed"]
            overall_accuracy = np.mean([s["accuracy"] for s in stats.values()]) if stats else 0.0

            print(f"\n{'='*10} {engine.upper()} SUMMARY {'='*10}")
            print(f"Files processed: {files}")
            print(f"Overall average accuracy: {overall_accuracy:.4f}")
            print(f"{'Field':<12} | {'Accuracy':<8} | {'Correct %':<10} | {'Min':<6} | {'Max':<6} | {'Median':<6}")
            print("-"*70)
            for field, s in stats.items():
                print(f"{field:<12} | {s['accuracy']:<8.4f} | {s['correct_percentage']:<10.2f} | "
                      f"{s['accuracy_min']:<6.4f} | {s['accuracy_max']:<6.4f} | {s['accuracy_median']:<6.4f}")

        print(f"\n{'='*10} COMPARATIVE ANALYSIS {'='*10}")
        comparison = self.results.get("comparison", {})
        print(f"{'Field':<12} | {'Tesseract':<10} | {'PaddleOCR':<10} | {'Diff':<8} | {'Improvement':<12} | {'Winner':<10}")
        print("-"*80)
        for field, c in comparison.items():
            improvement_str = f"+{c['percentage_improvement']:.2f}%" if c['percentage_improvement'] > 0 else f"{c['percentage_improvement']:.2f}%"
            print(f"{field:<12} | {c['tesseract_accuracy']:<10.4f} | {c['paddle_accuracy']:<10.4f} | "
                  f"{c['absolute_difference']:<8.4f} | {improvement_str:<12} | {c['winner']:<10}")

    def run_statistical_tests(self) -> Dict:
        """Run statistical tests for all fields."""
        stats_results = {}
        for field in FIELDS:
            tess_scores = [r["similarity"] for r in self.results["tesseract"]["field_results"][field]]
            paddle_scores = [r["similarity"] for r in self.results["paddle"]["field_results"][field]]

            stats_results[field] = {
                "paired_t_test": paired_t_test(tess_scores, paddle_scores),
                "cohens_d": cohens_d(tess_scores, paddle_scores),
                "mann_whitney_u": mann_whitney_u(tess_scores, paddle_scores)
            }

            # Print summary
            print(f"\nField: {field.upper()}")
            print(f"  Paired t-test: t={stats_results[field]['paired_t_test']['t_statistic']}, "
                  f"p={stats_results[field]['paired_t_test']['p_value']}, "
                  f"sig_0.05={stats_results[field]['paired_t_test']['significant_0.05']}")
            print(f"  Cohen's d: {stats_results[field]['cohens_d']:.4f}")
            print(f"  Mann-Whitney U: U={stats_results[field]['mann_whitney_u']['u_statistic']}, "
                  f"p={stats_results[field]['mann_whitney_u']['p_value']}, "
                  f"sig_0.05={stats_results[field]['mann_whitney_u']['significant_0.05']}")

        # Save to JSON
        output_path = "evaluation_results_statistics.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_results, f, indent=2)
        print(f"\n✅ Statistical test results saved to {output_path}")
        return stats_results

    def save_results(self, output_path: str = "evaluation_results_detailed.json"):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")


# ================================================================================  
# MAIN EXECUTION  
# ================================================================================  

if __name__ == "__main__":
    evaluator = OCRPerformanceEvaluator(TESSERACT_OCR_DIR, PADDLE_OCR_DIR, GROUND_TRUTH_DIR)
    evaluator.evaluate_all()
    evaluator.print_summary()
    evaluator.save_results()
    evaluator.run_statistical_tests()
