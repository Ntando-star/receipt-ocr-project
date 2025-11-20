# analysis/ensemble_regex/evaluate_ocr_performance.py
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
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ensemble_regex_extractor import EnsembleRegexExtractor

# ================================================================================  
# CONFIGURATION  
# ================================================================================  

TESSERACT_OCR_DIR = "data/ocr_output_tesseract_test"
PADDLE_OCR_DIR = "data/ocr_output_paddle_test"
GROUND_TRUTH_DIR = "data/raw/ground_truth"

FIELDS = ["company", "address", "date", "total"]

MODEL_DIRS = {
    "tesseract": Path("data/models/ensemble_tesseract"),
    "paddle": Path("data/models/ensemble_paddle")
}

OUTPUT_RESULTS_FILE = "evaluation_results_detailed_ensemble.json"
OUTPUT_STATS_FILE = "evaluation_results_statistics_ensemble.json"

# ================================================================================  
# HELPER FUNCTIONS  
# ================================================================================  

def string_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 1.0 if (not s1 and not s2) else 0.0
    return SequenceMatcher(None, s1.lower().strip(), s2.lower().strip()).ratio()


def load_ocr_text(ocr_dir: str, filename: str) -> str:
    path = Path(ocr_dir) / filename
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def load_ground_truth(filename: str) -> Dict:
    path = Path(GROUND_TRUTH_DIR) / filename
    if path.exists():
        try:
            data = json.load(path.open(encoding="utf-8"))
            return {field: data.get(field) for field in FIELDS}
        except json.JSONDecodeError:
            pass
    return {field: None for field in FIELDS}


def paired_t_test(scores1: List[float], scores2: List[float]) -> Dict:
    min_len = min(len(scores1), len(scores2))
    t_stat, p_value = stats.ttest_rel(scores1[:min_len], scores2[:min_len])
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_0.05": bool(p_value < 0.05),
        "significant_0.01": bool(p_value < 0.01)
    }


def mann_whitney_u(scores1: List[float], scores2: List[float]) -> Dict:
    min_len = min(len(scores1), len(scores2))
    u_stat, p_value = stats.mannwhitneyu(scores1[:min_len], scores2[:min_len])
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
# ENSEMBLE HELPER FUNCTIONS  
# ================================================================================  

def load_ensemble_extractors() -> Dict[str, Dict[str, EnsembleRegexExtractor]]:
    extractors = {"tesseract": {}, "paddle": {}}
    for engine in ["tesseract", "paddle"]:
        for field in FIELDS:
            model_path = MODEL_DIRS[engine] / f"{field}_ensemble.json"
            extractor = EnsembleRegexExtractor(field_name=field)
            if model_path.exists():
                extractor.load_model(str(model_path))
            extractors[engine][field] = extractor
    return extractors


def hybrid_extract(ocr_text: str, extractor: EnsembleRegexExtractor, field: str) -> str:
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


# ================================================================================  
# MAIN EVALUATOR  
# ================================================================================  

class EnsembleOCRPerformanceEvaluator:
    def __init__(self):
        self.extractors = load_ensemble_extractors()
        self.results = {"tesseract": {}, "paddle": {}}

    def evaluate_all(self) -> Dict:
        for engine, ocr_dir in [("tesseract", TESSERACT_OCR_DIR), ("paddle", PADDLE_OCR_DIR)]:
            # Only consider files that exist in the OCR test folder
            ocr_test_files = sorted([f.name for f in Path(ocr_dir).glob("*.txt")])
            field_results = {field: [] for field in FIELDS}

            for filename in ocr_test_files:
                ocr_text = load_ocr_text(ocr_dir, filename)
                gt_data = load_ground_truth(filename)
                if not ocr_text or not any(gt_data.values()):
                    continue
                for field in FIELDS:
                    pred = hybrid_extract(ocr_text, self.extractors[engine][field], field)
                    similarity = string_similarity(pred, gt_data.get(field, ""))
                    field_results[field].append({"similarity": similarity, "correct": similarity >= 0.8})

            stats = self._compute_field_statistics(field_results)
            self.results[engine]["field_results"] = field_results
            self.results[engine]["statistics"] = stats
            self.results[engine]["files_processed"] = len(ocr_test_files)  # correct count now

        self.results["comparison"] = self._compute_comparison_statistics()
        return self.results


    def _compute_field_statistics(self, field_results: Dict[str, List[Dict]]) -> Dict:
        stats_dict = {}
        for field, results in field_results.items():
            if not results:
                stats_dict[field] = self._empty_stats()
                continue
            similarities = [r["similarity"] for r in results]
            correct_count = sum(r["correct"] for r in results)
            stats_dict[field] = {
                "accuracy": round(np.mean(similarities), 4),
                "accuracy_std": round(np.std(similarities), 4),
                "accuracy_min": round(np.min(similarities), 4),
                "accuracy_max": round(np.max(similarities), 4),
                "accuracy_median": round(np.median(similarities), 4),
                "correct_count": correct_count,
                "correct_percentage": round(correct_count / len(results) * 100, 2),
                "incorrect_count": len(results) - correct_count,
                "n_samples": len(results),
                "iqr": list(np.percentile(similarities, [25, 75]))
            }
        return stats_dict

    def _empty_stats(self) -> Dict:
        return {k: 0.0 for k in ["accuracy", "accuracy_std", "accuracy_min", "accuracy_max",
                                 "accuracy_median", "correct_count", "correct_percentage",
                                 "incorrect_count", "n_samples", "iqr"]}

    def _compute_comparison_statistics(self) -> Dict:
        comparison = {}
        for field in FIELDS:
            t = self.results["tesseract"]["statistics"].get(field, self._empty_stats())
            p = self.results["paddle"]["statistics"].get(field, self._empty_stats())
            abs_diff = p["accuracy"] - t["accuracy"]
            pct_improvement = (abs_diff / t["accuracy"] * 100) if t["accuracy"] > 0 else 0
            winner = "PaddleOCR" if p["accuracy"] > t["accuracy"] else ("Tesseract" if t["accuracy"] > p["accuracy"] else "Tie")
            comparison[field] = {
                "tesseract_accuracy": round(t["accuracy"], 4),
                "paddle_accuracy": round(p["accuracy"], 4),
                "absolute_difference": round(abs_diff, 4),
                "percentage_improvement": round(pct_improvement, 2),
                "winner": winner,
                "correct_count_difference": p["correct_count"] - t["correct_count"]
            }
        return comparison

    def print_summary(self):
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

    def run_statistical_tests(self):
        stats_results = {}
        for field in FIELDS:
            t_scores = [r["similarity"] for r in self.results["tesseract"]["field_results"][field]]
            p_scores = [r["similarity"] for r in self.results["paddle"]["field_results"][field]]
            stats_results[field] = {
                "paired_t_test": paired_t_test(t_scores, p_scores),
                "cohens_d": cohens_d(t_scores, p_scores),
                "mann_whitney_u": mann_whitney_u(t_scores, p_scores)
            }
        with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats_results, f, indent=2)
        print(f"\n✅ Statistical test results saved to {OUTPUT_STATS_FILE}")
        return stats_results

    def save_results(self, output_path: str = OUTPUT_RESULTS_FILE):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")


# ================================================================================  
# MAIN EXECUTION  
# ================================================================================  

if __name__ == "__main__":
    evaluator = EnsembleOCRPerformanceEvaluator()
    evaluator.evaluate_all()
    evaluator.print_summary()
    evaluator.save_results()
    evaluator.run_statistical_tests()
