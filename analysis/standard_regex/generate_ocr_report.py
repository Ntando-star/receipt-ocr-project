# generate_ocr_report.py
import json
import os
from datetime import datetime
import numpy as np

FIELDS = ["company", "address", "date", "total"]

def generate_markdown_report(results_file: str = "evaluation_results_detailed.json",
                             stats_file: str = "statistical_testing_results.json") -> str:
    """Generate a markdown report for research write-up."""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    stats = {}
    if os.path.exists(stats_file):
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    
    # Safe metadata fallback
    metadata = results.get("evaluation_metadata", {
        "tesseract_files_processed": results.get("tesseract", {}).get("files_processed", 0),
        "paddle_files_processed": results.get("paddle", {}).get("files_processed", 0),
        "fields": FIELDS
    })

    report = f"""
# OCR Performance Evaluation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report presents a comprehensive statistical evaluation comparing **Tesseract OCR** and **PaddleOCR**   
on the receipt OCR dataset. Performance is measured across four key fields: company name, address,   
date, and total amount.

### Overall Performance

| Metric | Tesseract | PaddleOCR | Difference |
|--------|-----------|-----------|------------|
| Average Accuracy | {np.mean([results['tesseract']['statistics'][f]['accuracy'] for f in FIELDS]):.4f} | {np.mean([results['paddle']['statistics'][f]['accuracy'] for f in FIELDS]):.4f} | {(np.mean([results['paddle']['statistics'][f]['accuracy'] for f in FIELDS]) - np.mean([results['tesseract']['statistics'][f]['accuracy'] for f in FIELDS])):.4f} |
| Files Processed | {metadata['tesseract_files_processed']} | {metadata['paddle_files_processed']} | - |

---

## Detailed Results by Field

"""
    for field in FIELDS:
        tess = results['tesseract']['statistics'].get(field, {})
        paddle = results['paddle']['statistics'].get(field, {})
        comp = results.get('comparison', {}).get(field, {})
        
        # Calculate IQR safely
        tess_iqr = np.array(tess.get("iqr", []))
        paddle_iqr = np.array(paddle.get("iqr", []))
        
        report += f"""
### {field.upper()}

**Tesseract Performance:**
- Mean Accuracy: {tess.get('accuracy', 0.0):.4f}
- Standard Deviation: {tess.get('accuracy_std', 0.0):.4f}
- Correct Extractions: {tess.get('correct_count', 0)}/{tess.get('n_samples', 0)} ({tess.get('correct_percentage', 0.0):.2f}%)
- Range: [{tess.get('accuracy_min', 0.0):.4f}, {tess.get('accuracy_max', 0.0):.4f}]
- Median: {tess.get('accuracy_median', 0.0):.4f}

**PaddleOCR Performance:**
- Mean Accuracy: {paddle.get('accuracy', 0.0):.4f}
- Standard Deviation: {paddle.get('accuracy_std', 0.0):.4f}
- Correct Extractions: {paddle.get('correct_count', 0)}/{paddle.get('n_samples', 0)} ({paddle.get('correct_percentage', 0.0):.2f}%)
- Range: [{paddle.get('accuracy_min', 0.0):.4f}, {paddle.get('accuracy_max', 0.0):.4f}]
- Median: {paddle.get('accuracy_median', 0.0):.4f}

**Comparative Analysis:**
- Accuracy Difference: {comp.get('absolute_difference', 0.0):+.4f}
- Percentage Improvement: {comp.get('percentage_improvement', 0.0):+.2f}%
- Winner: **{comp.get('winner', 'Tie')}**
- Correct Count Difference: {comp.get('correct_count_difference', 0):+d}
"""
        if field in stats:
            field_stats = stats[field]
            report += f"""**Statistical Significance:**
- Paired t-test p-value: {field_stats['paired_t_test']['p_value']:.6f}
  - Significant (α=0.05): {'Yes' if field_stats['paired_t_test']['significant_0.05'] else 'No'}
- Mann-Whitney U p-value: {field_stats['mann_whitney_u']['p_value']:.6f}
  - Significant (α=0.05): {'Yes' if field_stats['mann_whitney_u']['significant_0.05'] else 'No'}
- Cohen's d (Effect Size): {field_stats['cohens_d']:.4f}
  - Interpretation: {get_effect_size_interpretation(field_stats['cohens_d'])}

"""

    report += f"""
---

## Interpretation & Conclusions

### Key Findings

1. **Field-by-Field Analysis**: The performance comparison reveals distinct patterns across different   
   receipt elements, suggesting that OCR engine selection should consider the specific field being extracted.

2. **Statistical Significance**: P-values from paired t-tests indicate whether performance differences   
   are statistically significant or due to random variation.

3. **Effect Sizes**: Cohen's d values quantify the practical significance of any observed differences.
   - Values < 0.2 indicate negligible practical difference
   - Values 0.2-0.5 indicate small practical difference
   - Values 0.5-0.8 indicate medium practical difference
   - Values > 0.8 indicate large practical difference

### Recommendations

- **Use PaddleOCR if**: Superior performance is required for critical fields (particularly where p-value < 0.05 and effect size is large)  
- **Use Tesseract if**: Lower computational overhead is a concern or performance is comparable across fields  
- **Hybrid Approach**: Consider implementing an ensemble method that selects the best extraction from both engines based on confidence scores

---

## Methodology

**Evaluation Criteria:**
- Similarity Threshold: 0.8 (string similarity for correctness)
- Fields Evaluated: {', '.join(metadata['fields'])}
- Statistical Tests: Paired t-test, Mann-Whitney U test, Cohen's d
- Total Samples Evaluated: {metadata['tesseract_files_processed']} (Tesseract), {metadata['paddle_files_processed']} (PaddleOCR)

---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    return report


def get_effect_size_interpretation(cohens_d: float) -> str:
    """Interpret Cohen's d value."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


if __name__ == "__main__":
    report = generate_markdown_report()
    
    # Save report
    with open("OCR_PERFORMANCE_REPORT.md", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Report saved to OCR_PERFORMANCE_REPORT.md")
    print("\nReport Preview:")
    print(report)
