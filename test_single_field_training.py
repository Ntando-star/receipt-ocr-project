import logging
import sys
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np

# Ensure imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.data_loader import DataLoader
from src.ensemble_regex_extractor import EnsembleRegexExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def string_similarity(a, b):
    if not a or not b:
        return 1.0 if not a and not b else 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def test_single_field_with_validation():

    loader = DataLoader(config)
    train_data, val_data, test_data = loader.load_paddle_data()

    field = "date"

    # Extract valid samples for training
    def filter_samples(dataset):
        return [
            {
                "ocr_text": s.ocr_text,
                "ground_truth": str(s.ground_truth.get(field)).strip(),
                "filename": s.filename
            }
            for s in dataset
            if s.ground_truth.get(field) not in (None, "", "None")
        ]

    train_samples = filter_samples(train_data)
    val_samples = filter_samples(val_data)
    test_samples = filter_samples(test_data)

    print(f"Training Samples: {len(train_samples)}")
    print(f"Validation Samples: {len(val_samples)}")
    print(f"Test Samples: {len(test_samples)}\n")

    if len(train_samples) == 0:
        print("❌ No training samples found!")
        return

    extractor = EnsembleRegexExtractor(
        field_name=field,
        n_patterns=config.ENSEMBLE_N_PATTERNS,
        min_accuracy=config.ENSEMBLE_MIN_ACCURACY
    )

    learning_results = extractor.learn_ensemble(
        training_data=train_samples,
        validation_data=val_samples,
        variance_threshold=config.ENSEMBLE_VARIANCE_THRESHOLD
    )

    def evaluate(samples, name):
        sims = []
        for s in samples:
            extraction = extractor.extract(s["ocr_text"])
            pred = extraction.get("value", "") or ""
            sims.append(string_similarity(pred, s["ground_truth"]))
        score = np.mean(sims) if sims else 0
        print(f"\n{name} Similarity Score: {score:.3f}")


    if not learning_results.get("success", False):
        print("❌ Regex learning failed")
        return

    print(f"\n📌 Learned Patterns for field '{field}':")
    for p in learning_results["patterns"]:
        print(f"  - Pattern: {p['pattern']}")
        print(f"    Accuracy: {p['accuracy']:.3f}")
        print(f"    Confidence: {p['confidence']:.3f}")
        # Evaluate on validation + test sets
    

    # Evaluate on validation + test
    def evaluate(samples, name):
        sims = []
        for s in samples:
            extraction = extractor.extract_with_confidence(s["ocr_text"])  # <-- change here
            pred = extraction.get("value", "") or ""
            sims.append(string_similarity(pred, s["ground_truth"]))
        score = np.mean(sims) if sims else 0
        print(f"\n{name} Similarity Score: {score:.3f}")

    # Evaluate on validation + test sets
    evaluate(val_samples, "Validation")
    evaluate(test_samples, "Test")

    if not learning_results.get("success", False):
        print("❌ Regex learning failed")
        return

    print(f"\n📌 Learned Patterns for field '{field}':")
    for p in learning_results["patterns"]:
        print(f"  - Pattern: {p['pattern']}")
        print(f"    Accuracy: {p['accuracy']:.3f}")
        print(f"    Confidence: {p['confidence']:.3f}")

    
if __name__ == "__main__":
    test_single_field_with_validation()
