# src/data_loader.py  
import os  
import json  
import logging  
from pathlib import Path  
from typing import Dict, List, Tuple  
from dataclasses import dataclass  
  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
@dataclass  
class DataSample:  
    """Single training/test sample"""  
    filename: str  
    ocr_text: str  
    ground_truth: Dict[str, str]  
    split: str  # 'train', 'val', 'test'  
      
class DataLoader:  
    """Load OCR outputs with ground truth annotations"""  
      
    def __init__(self, config):  
        self.config = config  
        self.fields = config.FIELDS  
      
    def load_split(self,   
               ocr_dir: Path,   
               gt_dir: Path,   
               split_name: str) -> List[DataSample]:
        """Load all samples for a split with stricter filtering"""
        samples = []

        ocr_files = sorted([f for f in os.listdir(ocr_dir) if f.endswith('.txt')])
        logger.info(f"Loading {split_name} set from {ocr_dir}")
        logger.info(f"Found {len(ocr_files)} OCR files")

        for ocr_file in ocr_files:
            try:
                # Load OCR text
                ocr_path = ocr_dir / ocr_file
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read().strip()

                # Skip empty or too short OCR
                if not ocr_text or len(ocr_text) < 5:
                    logger.warning(f"Skipping empty/short OCR file: {ocr_file}")
                    continue

                # Load ground truth
                gt_path = gt_dir / ocr_file
                if not gt_path.exists():
                    logger.warning(f"Missing GT for {ocr_file}")
                    continue

                with open(gt_path, 'r', encoding='utf-8') as f:
                    try:
                        gt_data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Corrupted GT JSON: {ocr_file}")
                        continue

                # Validate ground truth
                if not isinstance(gt_data, dict):
                    logger.warning(f"Invalid GT format in {ocr_file}")
                    continue

                gt = {field: gt_data.get(field, "") for field in self.fields}

                # Skip if all GT fields are empty
                if all(not v.strip() for v in gt.values()):
                    logger.warning(f"All GT fields empty: {ocr_file}")
                    continue

                sample = DataSample(
                    filename=ocr_file,
                    ocr_text=ocr_text,
                    ground_truth=gt,
                    split=split_name
                )
                samples.append(sample)

            except Exception as e:
                logger.error(f"Error loading {ocr_file}: {e}")
                continue

        logger.info(f"✅ Loaded {len(samples)} samples for {split_name}")
        return samples

      
    def load_tesseract_data(self) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:  
        """Load Tesseract train/val/test"""  
        train = self.load_split(  
            self.config.TRAIN_TESSERACT_DIR,  
            self.config.GROUND_TRUTH_DIR,  
            'train'  
        )  
        val = self.load_split(  
            self.config.VAL_TESSERACT_DIR,  
            self.config.GROUND_TRUTH_DIR,  
            'val'  
        )  
        test = self.load_split(  
            self.config.TEST_TESSERACT_DIR,  
            self.config.GROUND_TRUTH_DIR,  
            'test'  
        )  
        return train, val, test  
      
    def load_paddle_data(self) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:  
        """Load PaddleOCR train/val/test"""  
        train = self.load_split(  
            self.config.TRAIN_PADDLE_DIR,  
            self.config.GROUND_TRUTH_DIR,  
            'train'  
        )  
        val = self.load_split(  
            self.config.VAL_PADDLE_DIR,  
            self.config.GROUND_TRUTH_DIR,  
            'val'  
        )  
        test = self.load_split(  
            self.config.TEST_PADDLE_DIR,  
            self.config.GROUND_TRUTH_DIR,  
            'test'  
        )  
        return train, val, test  