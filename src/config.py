# src/config.py  
import os  
from pathlib import Path  
  
class Config:  
    # Base paths  
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  
    DATA_DIR = PROJECT_ROOT / "data"  
      
    # Raw data  
    TRAIN_RAW_DIR = DATA_DIR / "raw" / "training_data"  
    VAL_RAW_DIR = DATA_DIR / "raw" / "validation_data"  
    TEST_RAW_DIR = DATA_DIR / "raw" / "test_data"  
    GROUND_TRUTH_DIR = DATA_DIR / "raw" / "ground_truth"  
      
    # OCR outputs - TRAINING  
    TRAIN_TESSERACT_DIR = DATA_DIR / "ocr_output_tesseract"  
    TRAIN_PADDLE_DIR = DATA_DIR / "ocr_output_paddle"  
      
    # OCR outputs - VALIDATION  
    VAL_TESSERACT_DIR = DATA_DIR / "ocr_output_tesseract_val"  
    VAL_PADDLE_DIR = DATA_DIR / "ocr_output_paddle_val"  
      
    # OCR outputs - TEST  
    TEST_TESSERACT_DIR = DATA_DIR / "ocr_output_tesseract_test"  
    TEST_PADDLE_DIR = DATA_DIR / "ocr_output_paddle_test"  
      
    # Models  
    ENSEMBLE_MODELS_DIR = DATA_DIR / "models" / "ensemble"  
    BASELINE_MODELS_DIR = DATA_DIR / "models" / "baseline"  
      
    # Results  
    RESULTS_DIR = PROJECT_ROOT / "results"  
    MLRUNS_DIR = PROJECT_ROOT / "mlruns"  
      
    # Ensemble settings  
    ENSEMBLE_N_PATTERNS = 5  
    ENSEMBLE_MIN_ACCURACY = 0.6  
    ENSEMBLE_VARIANCE_THRESHOLD = 0.3  
      
    # Evaluation  
    SIMILARITY_THRESHOLD = 0.8  
      
    # Fields  
    FIELDS = ["company", "address", "date", "total"]  
      
    # Create directories  
    @classmethod  
    def setup_directories(cls):  
        for attr_name in dir(cls):  
            attr = getattr(cls, attr_name)  
            if isinstance(attr, Path) and attr_name.endswith("_DIR"):  
                attr.mkdir(parents=True, exist_ok=True)  
  
config = Config()  
config.setup_directories()  