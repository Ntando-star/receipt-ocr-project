# src/statistical_tests.py  
import numpy as np  
from scipy import stats  
import logging  
from typing import Dict, List, Tuple  
  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
class StatisticalTester:  
    """Comprehensive statistical testing"""  
      
    @staticmethod  
    def paired_t_test(group1: List[float], group2: List[float]) -> Dict:  
        """Paired t-test for dependent samples"""  
        if len(group1) != len(group2):  
            min_len = min(len(group1), len(group2))  
            group1 = group1[:min_len]  
            group2 = group2[:min_len]  
          
        t_stat, p_value = stats.ttest_rel(group1, group2)  
          
        return {  
            "test_name": "Paired t-test",  
            "t_statistic": round(float(t_stat), 4),  
            "p_value": round(float(p_value), 6),  
            "significant_at_0.05": bool(p_value < 0.05),  
            "significant_at_0.01": bool(p_value < 0.01),  
            "mean_diff": round(np.mean(np.array(group2) - np.array(group1)), 4)  
        }  
      
    @staticmethod  
    def mann_whitney_u(group1: List[float], group2: List[float]) -> Dict:  
        """Mann-Whitney U test (non-parametric alternative)"""  
        if len(group1) != len(group2):  
            min_len = min(len(group1), len(group2))  
            group1 = group1[:min_len]  
            group2 = group2[:min_len]  
          
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')  
          
        return {  
            "test_name": "Mann-Whitney U",  
            "u_statistic": round(float(u_stat), 4),  
            "p_value": round(float(p_value), 6),  
            "significant_at_0.05": bool(p_value < 0.05),  
            "significant_at_0.01": bool(p_value < 0.01)  
        }  
      
    @staticmethod  
    def cohens_d(group1: List[float], group2: List[float]) -> Dict:  
        """Cohen's d effect size"""  
        if len(group1) != len(group2):  
            min_len = min(len(group1), len(group2))  
            group1 = group1[:min_len]  
            group2 = group2[:min_len]  
          
        group1 = np.array(group1)  
        group2 = np.array(group2)  
          
        mean_diff = np.mean(group2) - np.mean(group1)  
        n1, n2 = len(group1), len(group2)  
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)  
          
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))  
          
        if pooled_std == 0:  
            cohens_d = 0.0  
        else:  
            cohens_d = float(mean_diff / pooled_std)  
          
        # Interpretation  
        abs_d = abs(cohens_d)  
        if abs_d < 0.2:  
            interpretation = "Negligible"  
        elif abs_d < 0.5:  
            interpretation = "Small"  
        elif abs_d < 0.8:  
            interpretation = "Medium"  
        else:  
            interpretation = "Large"  
          
        return {  
            "test_name": "Cohen's d",  
            "cohens_d": round(cohens_d, 4),  
            "interpretation": interpretation,  
            "group1_mean": round(np.mean(group1), 4),  
            "group2_mean": round(np.mean(group2), 4),  
            "mean_difference": round(mean_diff, 4)  
        }  
      
    @staticmethod  
    def normality_test(group: List[float]) -> Dict:  
        """Shapiro-Wilk normality test"""  
        stat, p_value = stats.shapiro(group)  
          
        return {  
            "test_name": "Shapiro-Wilk",  
            "statistic": round(float(stat), 4),  
            "p_value": round(float(p_value), 6),  
            "is_normal_at_0.05": bool(p_value > 0.05)  
        }  
      
    @classmethod  
    def run_comprehensive_tests(cls,   
                               baseline_scores: List[float],  
                               ensemble_scores: List[float],  
                               field_name: str = "field") -> Dict:  
        """Run all tests and return comprehensive results"""  
        if not baseline_scores or not ensemble_scores:
            logger.warning(f"No valid scores for field '{field_name}'. Skipping statistical tests.")
            return {
                "field": field_name,
                "baseline_stats": None,
                "ensemble_stats": None,
                "normality_baseline": None,
                "normality_ensemble": None,
                "paired_t_test": None,
                "mann_whitney_u": None,
                "cohens_d": None
            }
        logger.info(f"\n{'='*70}")  
        logger.info(f"Statistical Analysis: {field_name.upper()}")  
        logger.info(f"{'='*70}")  
        logger.info(f"Baseline: n={len(baseline_scores)}, mean={np.mean(baseline_scores):.4f}")  
        logger.info(f"Ensemble: n={len(ensemble_scores)}, mean={np.mean(ensemble_scores):.4f}")  
          
        results = {  
            "field": field_name,  
            "baseline_stats": {  
                "n": len(baseline_scores),  
                "mean": round(np.mean(baseline_scores), 4),  
                "std": round(np.std(baseline_scores), 4),  
                "min": round(np.min(baseline_scores), 4),  
                "max": round(np.max(baseline_scores), 4),  
                "median": round(np.median(baseline_scores), 4)  
            },  
            "ensemble_stats": {  
                "n": len(ensemble_scores),  
                "mean": round(np.mean(ensemble_scores), 4),  
                "std": round(np.std(ensemble_scores), 4),  
                "min": round(np.min(ensemble_scores), 4),  
                "max": round(np.max(ensemble_scores), 4),  
                "median": round(np.median(ensemble_scores), 4)  
            },  
            "normality_baseline": cls.normality_test(baseline_scores),  
            "normality_ensemble": cls.normality_test(ensemble_scores),  
            "paired_t_test": cls.paired_t_test(baseline_scores, ensemble_scores),  
            "mann_whitney_u": cls.mann_whitney_u(baseline_scores, ensemble_scores),  
            "cohens_d": cls.cohens_d(baseline_scores, ensemble_scores)  
        }  
          
        logger.info(f"\nNormality (Shapiro-Wilk):")  
        logger.info(f"  Baseline: p={results['normality_baseline']['p_value']:.6f}, "  
                   f"normal={results['normality_baseline']['is_normal_at_0.05']}")  
        logger.info(f"  Ensemble: p={results['normality_ensemble']['p_value']:.6f}, "  
                   f"normal={results['normality_ensemble']['is_normal_at_0.05']}")  
          
        logger.info(f"\nPaired t-test:")  
        logger.info(f"  t={results['paired_t_test']['t_statistic']}, "  
                   f"p={results['paired_t_test']['p_value']}, "  
                   f"sig={results['paired_t_test']['significant_at_0.05']}")  
          
        logger.info(f"\nMann-Whitney U:")  
        logger.info(f"  U={results['mann_whitney_u']['u_statistic']}, "  
                   f"p={results['mann_whitney_u']['p_value']}, "  
                   f"sig={results['mann_whitney_u']['significant_at_0.05']}")  
          
        logger.info(f"\nCohen's d:")  
        logger.info(f"  d={results['cohens_d']['cohens_d']}, "  
                   f"interpretation={results['cohens_d']['interpretation']}")  
          
        logger.info(f"{'='*70}\n")  
          
        return results  