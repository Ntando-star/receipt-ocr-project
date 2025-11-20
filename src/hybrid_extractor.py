# src/hybrid_extractor.py
import re
import json
from pathlib import Path
from src.ensemble_regex_extractor import EnsembleRegexExtractor

# -------------------------
# Hand-Coded Regex Rules
# -------------------------
def extract_company(text):
    patterns = [
        r"[A-Z0-9&\(\)\- ]+SDN BHD",
        r"[A-Z0-9&\(\)\- ]+LTD",
        r"[A-Z0-9&\(\)\- ]+ENTERPRISE",
        r"[A-Z0-9&\(\)\- ]+TRADING"
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None

def extract_address(text):
    # naive approach: look for line with numbers and street keywords
    patterns = [
        r"\d{1,5}[A-Z]?[, ]+.*(JALAN|ROAD|STREET|TAMAN|NO\.)[, ]+.*",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()
    return None

def extract_date(text):
    patterns = [
        r"\d{1,2}/\d{1,2}/\d{2,4}",
        r"\d{4}-\d{1,2}-\d{1,2}"
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(0).strip()
    return None

def extract_total(text):
    patterns = [
        r"Total[: ]+[\d.,]+",
        r"\bRM[: ]*[\d.,]+",
        r"[\d.,]+\s*(?:RM|USD|R)"
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            # extract only numeric part
            number = re.search(r"[\d.,]+", match.group(0))
            if number:
                return number.group(0).replace(",", "").strip()
    return None

# Mapping
HAND_REGEX_FUNCS = {
    "company": extract_company,
    "address": extract_address,
    "date": extract_date,
    "total": extract_total
}

# -------------------------
# Hybrid Extractor Class
# -------------------------
class HybridExtractor:
    def __init__(self, field, ensemble_model_path=None):
        self.field = field
        self.ensemble = None
        if ensemble_model_path and Path(ensemble_model_path).exists():
            self.ensemble = EnsembleRegexExtractor(field)
            self.ensemble.load_model(ensemble_model_path)

    def extract(self, text):
        # 1️⃣ Try hand-coded regex first
        hand_value = HAND_REGEX_FUNCS[self.field](text)
        if hand_value:
            return {"value": hand_value, "source": "hand-coded"}

        # 2️⃣ Fall back to ensemble if available
        if self.ensemble:
            result = self.ensemble.extract_with_confidence(text)
            return {"value": result["value"], "source": "ensemble"}

        # 3️⃣ Nothing found
        return {"value": None, "source": None}

# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    # Load an OCR text file
    example_file = Path("data/ocr_output_tesseract_test/X51005442376.txt")
    text = example_file.read_text()

    for field in ["company", "address", "date", "total"]:
        hybrid = HybridExtractor(
            field,
            ensemble_model_path=f"data/models/ensemble_tesseract/{field}_ensemble.json"
        )
        result = hybrid.extract(text)
        print(f"{field.upper():<10} | {result['value']} | Source: {result['source']}")
