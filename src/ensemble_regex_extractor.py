# src/ensemble_regex_extractor.py
import re
from difflib import SequenceMatcher
from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)

class EnsembleRegexExtractor:

    def __init__(
        self,
        field_name: str,
        n_patterns: int = 5,
        min_accuracy: float = 0.4,
        pattern_complexity_penalty: float = 0.01,
        voting_threshold: float = 0.6
    ):
        self.field_name = field_name
        self.n_patterns = n_patterns
        self.min_accuracy = min_accuracy
        self.pattern_complexity_penalty = pattern_complexity_penalty
        self.voting_threshold = voting_threshold
        self.patterns = []

    # -----------------------
    # OCR text preprocessing
    # -----------------------
    @staticmethod
    def preprocess_ocr_text(text: str) -> str:
        """Normalize OCR text: unify whitespace, remove extra symbols, lowercase."""
        if not text:
            return ""
        text = text.replace('\n', ' ')  # join lines
        text = re.sub(r'[\r\t]', ' ', text)
        text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
        return text.strip()

    # -----------------------
    # Utility functions
    # -----------------------
    def _string_similarity(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 1.0 if (not s1 and not s2) else 0.0
        return SequenceMatcher(None, s1.lower().strip(), s2.lower().strip()).ratio()

    def _calculate_pattern_complexity(self, pattern: str) -> float:
        return len(pattern) + pattern.count("|") * 10 + pattern.count("[") * 5 + pattern.count("(?") * 3

    # -----------------------
    # Candidate pattern generation (robust)
    # -----------------------
    def _extract_company_patterns(self, ocr_text: str, gt_value: str) -> List[str]:
        patterns = []
        company_keywords = [
            'SDN BHD', 'LTD', 'ENTERPRISE', 'TRADING', 'PLT',
            'SHOP', 'STORE', 'DECO', 'MART', 'MARKET', 'CO'
        ]
        # Generic company pattern
        patterns.append(r'([A-Z][A-Z0-9\s\.\-&]{3,}(?:' + '|'.join(company_keywords) + r'))')
        # Add words from GT if available
        if gt_value:
            for word in gt_value.split()[:3]:
                if len(word) > 2:
                    patterns.append(rf'\b{re.escape(word)}\b')
        return patterns

    def _extract_address_patterns(self, ocr_text: str, gt_value: str) -> List[str]:
        patterns = []
        # Generic address pattern
        patterns.append(r'([0-9]{1,5}[\.,]?\s?[A-Z0-9\s,&/-]{5,100})')
        patterns.append(r'(?:NO|LOT)[\s\.]*[0-9A-Z,/-\s]+')
        patterns.append(r'\b\d{5}\b')  # postal code
        # Add words from GT if available
        if gt_value:
            for word in gt_value.split()[:2]:
                if len(word) > 2:
                    patterns.append(rf'\b{re.escape(word)}\b')
        return patterns

    def _extract_date_patterns(self, ocr_text: str, gt_value: str) -> List[str]:
        return [
            r'(\d{2}[/-]\d{2}[/-]\d{4})',      # 25/12/2018
            r'(\d{2}[/-]\d{2}[/-]\d{2})',      # 25/12/18
            r'(\d{1,2}\s[A-Z]{3}\s\d{4})',     # 25 DEC 2018
        ]

    def _extract_total_patterns(self, ocr_text: str, gt_value: str) -> List[str]:
        patterns = [
            r'(?:RM|\$)?\s*([0-9]+[.,][0-9]{2})',   # 9.00 or RM 9.00
            r'([0-9]+[.,][0-9]{2})\s*$',            # ending with number
        ]
        keywords = ["TOTAL", "GRAND TOTAL", "AMOUNT", "ROUNDED TOTAL"]
        for kw in keywords:
            if kw.lower() in ocr_text.lower():
                patterns.append(rf'(?i){kw}[:\s]*([0-9]+[.,][0-9]{{2}})')
        return patterns

    def _generate_candidates(self, training: List[Dict]) -> List[str]:
        candidates = set()
        extractors = {
            "date": self._extract_date_patterns,
            "total": self._extract_total_patterns,
            "company": self._extract_company_patterns,
            "address": self._extract_address_patterns
        }
        if self.field_name not in extractors:
            logger.warning(f"No pattern generator for field: {self.field_name}")
            return []
        extractor_func = extractors[self.field_name]
        for data in training:
            ocr_text = self.preprocess_ocr_text(data["ocr_text"])
            gt_value = data["ground_truth"]
            try:
                patterns = extractor_func(ocr_text, gt_value)
                for p in patterns:
                    if p:
                        candidates.add(p)
            except Exception as e:
                logger.debug(f"Error generating patterns: {e}")
                continue
        return list(candidates)

    # -----------------------
    # Main training function
    # -----------------------
    def learn_ensemble(self,
                       training_data: List[Dict],
                       validation_data: Optional[List[Dict]] = None,
                       variance_threshold: float = 0.3,
                       use_complexity_penalty: bool = True):
        if not training_data:
            return {"success": False, "error": "No training data provided"}

        selection_data = validation_data if validation_data else training_data
        candidates = self._generate_candidates(training_data)
        if not candidates:
            return {"success": False, "error": f"No candidate patterns for field '{self.field_name}'"}

        candidate_scores = {}
        for pattern in candidates:
            accuracies = []
            try:
                for data in selection_data:
                    ocr_text = self.preprocess_ocr_text(data["ocr_text"])
                    match = re.search(pattern, ocr_text, re.IGNORECASE)
                    if match:
                        extracted = match.group(1) if match.groups() else match.group(0)
                        accuracies.append(self._string_similarity(extracted, data["ground_truth"]))
                if accuracies:
                    candidate_scores[pattern] = {"accuracy": np.mean(accuracies), "accuracies": accuracies}
            except Exception as e:
                logger.debug(f"Error evaluating pattern {pattern[:30]}: {e}")
                continue

        filtered = {p: s for p, s in candidate_scores.items() if s["accuracy"] >= self.min_accuracy}
        if not filtered:
            filtered = dict(sorted(candidate_scores.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:self.n_patterns])

        scored = []
        for p, s in filtered.items():
            acc = s["accuracy"]
            var = np.var(s["accuracies"]) if len(s["accuracies"]) > 1 else 0
            cx = self._calculate_pattern_complexity(p)
            reg = acc - (var * variance_threshold) - (cx * self.pattern_complexity_penalty if use_complexity_penalty else 0)
            reg = max(0, min(1, reg))
            scored.append((p, acc, reg))

        scored.sort(key=lambda x: x[2], reverse=True)
        selected = scored[:self.n_patterns]

        self.patterns = [{"pattern": p, "accuracy": float(a), "regularized_accuracy": float(r), "weight": float(a), "confidence": float(r)}
                         for p, a, r in selected]

        avg_acc = float(np.mean([p["accuracy"] for p in self.patterns])) if self.patterns else 0.0
        return {"success": True, "patterns": self.patterns, "n_patterns": len(self.patterns),
                "average_accuracy": avg_acc, "total_candidates": len(candidates)}

    # -----------------------
    # Fine-tune, evaluate, extract, save/load (unchanged)
    # -----------------------
    def fine_tune(self, validation_data: List[Dict]):
        if not validation_data or not self.patterns:
            return
        refined_patterns = []
        for p in self.patterns:
            accuracies = []
            for data in validation_data:
                try:
                    ocr_text = self.preprocess_ocr_text(data["ocr_text"])
                    match = re.search(p["pattern"], ocr_text, re.IGNORECASE)
                    if match:
                        extracted = match.group(1) if match.groups() else match.group(0)
                        accuracies.append(self._string_similarity(extracted, data["ground_truth"]))
                except Exception:
                    continue
            avg_acc = np.mean(accuracies) if accuracies else 0.0
            if avg_acc >= self.min_accuracy:
                refined_patterns.append(p)
        if refined_patterns:
            self.patterns = refined_patterns

    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        if not test_data or not self.patterns:
            return {"exact_accuracy": 0.0, "avg_similarity": 0.0}

        exact_matches = 0
        similarities = []
        for data in test_data:
            result = self.extract_with_confidence(data["ocr_text"])
            predicted = result["value"] or ""
            gt = data["ground_truth"] or ""
            sim = self._string_similarity(predicted, gt)
            similarities.append(sim)
            if predicted.strip() == gt.strip():
                exact_matches += 1

        exact_acc = exact_matches / len(test_data)
        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        return {"exact_accuracy": exact_acc, "avg_similarity": avg_sim}

    def extract_with_confidence(self, ocr_text: str) -> dict:
        if not self.patterns:
            return {"value": None, "confidence": 0.0, "matches": [], "n_patterns": 0, "agreement": 0}

        ocr_text = self.preprocess_ocr_text(ocr_text)
        matches = []
        for p in self.patterns:
            pattern = p["pattern"]
            weight = p.get("weight", 1.0)
            try:
                m = re.search(pattern, ocr_text, re.IGNORECASE)
                if m:
                    value = m.group(1) if m.groups() else m.group(0)
                    matches.append({"value": value, "weight": weight})
            except Exception:
                continue

        if not matches:
            return {"value": None, "confidence": 0.0, "matches": [], "n_patterns": len(self.patterns), "agreement": 0}

        vote_counter = defaultdict(float)
        for m in matches:
            vote_counter[m["value"]] += m["weight"]

        best_value = max(vote_counter.items(), key=lambda x: x[1])[0]
        total_weight = sum(vote_counter.values())
        confidence = vote_counter[best_value] / total_weight if total_weight else 0.0
        agreement = sum(1 for m in matches if m["value"] == best_value) / len(matches) if matches else 0

        return {"value": best_value, "confidence": float(confidence), "matches": matches,
                "n_patterns": len(self.patterns), "agreement": float(agreement)}

    def extract(self, ocr_text: str) -> dict:
        result = self.extract_with_confidence(ocr_text)
        return {"value": result["value"]}

    def save_model(self, path: str):
        model_data = {
            "field_name": self.field_name,
            "patterns": self.patterns,
            "n_patterns": self.n_patterns,
            "min_accuracy": self.min_accuracy
        }
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> bool:
        try:
            with open(path, 'r') as f:
                model_data = json.load(f)
            self.patterns = model_data.get("patterns", [])
            self.field_name = model_data.get("field_name", self.field_name)
            logger.info(f"Model loaded from {path}: {len(self.patterns)} patterns")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
