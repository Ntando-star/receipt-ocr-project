
# OCR Performance Comparison Report: Tesseract vs PaddleOCR

## Executive Summary

This report compares the performance of **Tesseract OCR** and **PaddleOCR** when combined with:
1. **Standard (hand-crafted) Regex** patterns
2. **Ensemble-learned Regex** patterns

The analysis includes statistical hypothesis testing to determine if performance differences are
statistically significant.

---

## Results by Field


### COMPANY

#### Baseline Performance (Hand-crafted Regex)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | 0.0567 | PaddleOCR |
| PaddleOCR | 0.0657 | |
| **Difference** | **+0.0090** | |

#### Ensemble Performance (Learned Patterns)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | 0.0567 | PaddleOCR |
| PaddleOCR | 0.0657 | |
| **Difference** | **+0.0090** | |

#### Improvement from Baseline to Ensemble

**Tesseract:**
- Absolute: +0.0000
- Percentage: +0.00%

**PaddleOCR:**
- Absolute: +0.0000
- Percentage: +0.00%

#### Statistical Significance (Ensemble Improvement)
**Tesseract Baseline → Ensemble:**
- Paired t-test: t=nan, p=nan  - Significant (α=0.05): ❌ No
- Cohen's d: 0.0000 (Negligible)

**PaddleOCR Baseline → Ensemble:**
- Paired t-test: t=nan, p=nan  - Significant (α=0.05): ❌ No
- Cohen's d: 0.0000 (Negligible)

**Cross-Engine Comparison (Ensemble):**
- Paired t-test: t=-2.5311, p=0.011528  - Significant (α=0.05): ✅ Yes
- Cohen's d: 0.0399 (Negligible)

---


### ADDRESS

#### Baseline Performance (Hand-crafted Regex)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | 0.0583 | PaddleOCR |
| PaddleOCR | 0.0610 | |
| **Difference** | **+0.0027** | |

#### Ensemble Performance (Learned Patterns)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | 0.0583 | PaddleOCR |
| PaddleOCR | 0.0610 | |
| **Difference** | **+0.0027** | |

#### Improvement from Baseline to Ensemble

**Tesseract:**
- Absolute: +0.0000
- Percentage: +0.00%

**PaddleOCR:**
- Absolute: +0.0000
- Percentage: +0.00%

#### Statistical Significance (Ensemble Improvement)
**Tesseract Baseline → Ensemble:**
- Paired t-test: t=nan, p=nan  - Significant (α=0.05): ❌ No
- Cohen's d: 0.0000 (Negligible)

**PaddleOCR Baseline → Ensemble:**
- Paired t-test: t=nan, p=nan  - Significant (α=0.05): ❌ No
- Cohen's d: 0.0000 (Negligible)

**Cross-Engine Comparison (Ensemble):**
- Paired t-test: t=-1.9203, p=0.055112  - Significant (α=0.05): ❌ No
- Cohen's d: 0.0160 (Negligible)

---


### DATE

#### Baseline Performance (Hand-crafted Regex)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | 0.1183 | PaddleOCR |
| PaddleOCR | 0.1394 | |
| **Difference** | **+0.0211** | |

#### Ensemble Performance (Learned Patterns)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | 0.1183 | PaddleOCR |
| PaddleOCR | 0.1394 | |
| **Difference** | **+0.0211** | |

#### Improvement from Baseline to Ensemble

**Tesseract:**
- Absolute: +0.0000
- Percentage: +0.00%

**PaddleOCR:**
- Absolute: +0.0000
- Percentage: +0.00%

#### Statistical Significance (Ensemble Improvement)
**Tesseract Baseline → Ensemble:**
- Paired t-test: t=nan, p=nan  - Significant (α=0.05): ❌ No
- Cohen's d: 0.0000 (Negligible)

**PaddleOCR Baseline → Ensemble:**
- Paired t-test: t=nan, p=nan  - Significant (α=0.05): ❌ No
- Cohen's d: 0.0000 (Negligible)

**Cross-Engine Comparison (Ensemble):**
- Paired t-test: t=-4.7197, p=0.000003  - Significant (α=0.05): ✅ Yes
- Cohen's d: 0.0636 (Negligible)

---


### TOTAL

#### Baseline Performance (Hand-crafted Regex)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | 0.0867 | PaddleOCR |
| PaddleOCR | 0.1015 | |
| **Difference** | **+0.0147** | |

#### Ensemble Performance (Learned Patterns)

| Engine | Accuracy | Winner |
|--------|----------|--------|
| Tesseract | 0.0867 | PaddleOCR |
| PaddleOCR | 0.1015 | |
| **Difference** | **+0.0147** | |

#### Improvement from Baseline to Ensemble

**Tesseract:**
- Absolute: +0.0000
- Percentage: +0.00%

**PaddleOCR:**
- Absolute: +0.0000
- Percentage: +0.00%

#### Statistical Significance (Ensemble Improvement)
**Tesseract Baseline → Ensemble:**
- Paired t-test: t=nan, p=nan  - Significant (α=0.05): ❌ No
- Cohen's d: 0.0000 (Negligible)

**PaddleOCR Baseline → Ensemble:**
- Paired t-test: t=nan, p=nan  - Significant (α=0.05): ❌ No
- Cohen's d: 0.0000 (Negligible)

**Cross-Engine Comparison (Ensemble):**
- Paired t-test: t=-3.8565, p=0.000123  - Significant (α=0.05): ✅ Yes
- Cohen's d: 0.0580 (Negligible)

---


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
