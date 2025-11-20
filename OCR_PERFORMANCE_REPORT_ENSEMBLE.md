
# Ensemble OCR Performance Evaluation Report

**Date**: 2025-11-19 22:24:05

---

## Executive Summary

This report presents a comprehensive statistical evaluation comparing **Tesseract OCR** and **PaddleOCR**  
when used with ensemble-learned Regex patterns for information extraction. Performance is measured across four key fields: company name, address,  
date, and total amount.

### Overall Performance

| Metric | Tesseract | PaddleOCR | Difference |
|--------|-----------|-----------|------------|
| Average Accuracy | 0.5335 | 0.6048 | 0.0714 |
| Files Processed | 147 | 147 | - |

---

## Detailed Results by Field


### COMPANY

**Tesseract Performance:**
- Mean Accuracy: 0.3804
- Standard Deviation: 0.4328
- Correct Extractions: 45/145 (31.03%)
- Range: [0.0000, 1.0000]
- Median: 0.0000

**PaddleOCR Performance:**
- Mean Accuracy: 0.4346
- Standard Deviation: 0.4502
- Correct Extractions: 55/147 (37.41%)
- Range: [0.0000, 1.0000]
- Median: 0.4151

**Comparative Analysis:**
- Accuracy Difference: +0.0542
- Percentage Improvement: +14.25%
- Winner: **PaddleOCR**
- Correct Count Difference: +10
**Statistical Significance:**
- Paired t-test p-value: 0.257305
  - Significant (α=0.05): No
- Mann-Whitney U p-value: 0.293302
  - Significant (α=0.05): No
- Cohen's d (Effect Size): -0.1050
  - Interpretation: Negligible


### ADDRESS

**Tesseract Performance:**
- Mean Accuracy: 0.3846
- Standard Deviation: 0.2202
- Correct Extractions: 1/145 (0.69%)
- Range: [0.0000, 0.9495]
- Median: 0.4000

**PaddleOCR Performance:**
- Mean Accuracy: 0.3971
- Standard Deviation: 0.2245
- Correct Extractions: 1/147 (0.68%)
- Range: [0.0682, 0.9796]
- Median: 0.4270

**Comparative Analysis:**
- Accuracy Difference: +0.0125
- Percentage Improvement: +3.25%
- Winner: **PaddleOCR**
- Correct Count Difference: +0
**Statistical Significance:**
- Paired t-test p-value: 0.402604
  - Significant (α=0.05): No
- Mann-Whitney U p-value: 0.620828
  - Significant (α=0.05): No
- Cohen's d (Effect Size): -0.0679
  - Interpretation: Negligible


### DATE

**Tesseract Performance:**
- Mean Accuracy: 0.7939
- Standard Deviation: 0.3804
- Correct Extractions: 113/145 (77.93%)
- Range: [0.0000, 1.0000]
- Median: 1.0000

**PaddleOCR Performance:**
- Mean Accuracy: 0.9228
- Standard Deviation: 0.2442
- Correct Extractions: 134/147 (91.16%)
- Range: [0.0000, 1.0000]
- Median: 1.0000

**Comparative Analysis:**
- Accuracy Difference: +0.1289
- Percentage Improvement: +16.24%
- Winner: **PaddleOCR**
- Correct Count Difference: +21
**Statistical Significance:**
- Paired t-test p-value: 0.000151
  - Significant (α=0.05): Yes
- Mann-Whitney U p-value: 0.000173
  - Significant (α=0.05): Yes
- Cohen's d (Effect Size): -0.3977
  - Interpretation: Small


### TOTAL

**Tesseract Performance:**
- Mean Accuracy: 0.5751
- Standard Deviation: 0.3009
- Correct Extractions: 45/145 (31.03%)
- Range: [0.0000, 1.0000]
- Median: 0.5000

**PaddleOCR Performance:**
- Mean Accuracy: 0.6649
- Standard Deviation: 0.3172
- Correct Extractions: 68/147 (46.26%)
- Range: [0.0000, 1.0000]
- Median: 0.6667

**Comparative Analysis:**
- Accuracy Difference: +0.0898
- Percentage Improvement: +15.61%
- Winner: **PaddleOCR**
- Correct Count Difference: +23
**Statistical Significance:**
- Paired t-test p-value: 0.007119
  - Significant (α=0.05): Yes
- Mann-Whitney U p-value: 0.037060
  - Significant (α=0.05): Yes
- Cohen's d (Effect Size): -0.2745
  - Interpretation: Small


---

## Interpretation & Conclusions

### Recommendations

- **Use PaddleOCR if**: Superior performance is required for critical fields (p-value < 0.05 and effect size large)  
- **Use Tesseract if**: Lower computational overhead is a concern or performance is comparable  
- **Hybrid Approach**: Consider an ensemble method selecting the best extraction based on confidence scores

---

## Methodology

**Evaluation Criteria:**
- Similarity Threshold: 0.8 (string similarity for correctness)
- Fields Evaluated: company, address, date, total
- Statistical Tests: Paired t-test, Mann-Whitney U test, Cohen's d
- Total Samples Evaluated: 147 (Tesseract), 147 (PaddleOCR)

---

*Report generated: 2025-11-19 22:24:05*
