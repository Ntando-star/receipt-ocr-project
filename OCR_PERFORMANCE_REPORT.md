
# OCR Performance Evaluation Report

**Date**: 2025-11-19 21:46:29

---

## Executive Summary

This report presents a comprehensive statistical evaluation comparing **Tesseract OCR** and **PaddleOCR**   
on the receipt OCR dataset. Performance is measured across four key fields: company name, address,   
date, and total amount.

### Overall Performance

| Metric | Tesseract | PaddleOCR | Difference |
|--------|-----------|-----------|------------|
| Average Accuracy | 0.6885 | 0.6786 | -0.0100 |
| Files Processed | 145 | 147 | - |

---

## Detailed Results by Field


### COMPANY

**Tesseract Performance:**
- Mean Accuracy: 0.7494
- Standard Deviation: 0.2879
- Correct Extractions: 81/145 (55.86%)
- Range: [0.0526, 1.0000]
- Median: 0.8485

**PaddleOCR Performance:**
- Mean Accuracy: 0.7770
- Standard Deviation: 0.2829
- Correct Extractions: 87/147 (59.18%)
- Range: [0.0526, 1.0000]
- Median: 0.9259

**Comparative Analysis:**
- Accuracy Difference: +0.0276
- Percentage Improvement: +3.68%
- Winner: **PaddleOCR**
- Correct Count Difference: +0

### ADDRESS

**Tesseract Performance:**
- Mean Accuracy: 0.5687
- Standard Deviation: 0.1904
- Correct Extractions: 12/145 (8.28%)
- Range: [0.0000, 0.8295]
- Median: 0.5981

**PaddleOCR Performance:**
- Mean Accuracy: 0.6042
- Standard Deviation: 0.1939
- Correct Extractions: 20/147 (13.61%)
- Range: [0.0625, 0.8908]
- Median: 0.6360

**Comparative Analysis:**
- Accuracy Difference: +0.0355
- Percentage Improvement: +6.24%
- Winner: **PaddleOCR**
- Correct Count Difference: +0

### DATE

**Tesseract Performance:**
- Mean Accuracy: 0.7416
- Standard Deviation: 0.4171
- Correct Extractions: 105/145 (72.41%)
- Range: [0.0000, 1.0000]
- Median: 1.0000

**PaddleOCR Performance:**
- Mean Accuracy: 0.8473
- Standard Deviation: 0.3437
- Correct Extractions: 123/147 (83.67%)
- Range: [0.0000, 1.0000]
- Median: 1.0000

**Comparative Analysis:**
- Accuracy Difference: +0.1057
- Percentage Improvement: +14.25%
- Winner: **PaddleOCR**
- Correct Count Difference: +0

### TOTAL

**Tesseract Performance:**
- Mean Accuracy: 0.6945
- Standard Deviation: 0.3303
- Correct Extractions: 81/145 (55.86%)
- Range: [0.0000, 1.0000]
- Median: 0.8000

**PaddleOCR Performance:**
- Mean Accuracy: 0.4859
- Standard Deviation: 0.2985
- Correct Extractions: 36/147 (24.49%)
- Range: [0.0000, 1.0000]
- Median: 0.4000

**Comparative Analysis:**
- Accuracy Difference: -0.2086
- Percentage Improvement: -30.04%
- Winner: **Tesseract**
- Correct Count Difference: +0

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
- Fields Evaluated: company, address, date, total
- Statistical Tests: Paired t-test, Mann-Whitney U test, Cohen's d
- Total Samples Evaluated: 145 (Tesseract), 147 (PaddleOCR)

---

*Report generated: 2025-11-19 21:46:29*
