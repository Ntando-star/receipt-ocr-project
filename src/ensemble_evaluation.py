# src/ensemble_evaluation.py  
import os  
import json  
from pathlib import Path  
import logging  
from typing import List, Dict, Tuple  
from difflib import SequenceMatcher  
import numpy as np  
  
import mlflow  
  
from src.config import config  
from src.ensemble_regex_extractor import EnsembleRegexExtractor  
from src.extract_fields import extract_fields as extract_tesseract_baseline  
from src.extract_fields_paddle import extract_fields as extract_paddle_baseline  
  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
FIELDS = ["company", "address", "date", "total"]  
  
  
class EnsembleEvaluator:  
    """Evaluate ensemble extractor against baselines"""  
      
    def __init__(self, ocr_dir: str, gt_dir: str):  
        self.ocr_dir = ocr_dir  
        self.gt_dir = gt_dir  
        self.results = {}  
      
    def load_training_data(self, sample_size: int = None) -> Dict[str, List[Dict]]:  
        """  
        Load labeled data for training ensemble.  
          
        Ground truth files are in format:  
        {  
            "company": "BOOK TA .K (TAMAN DAYA) SDN BHD",  
            "date": "25/12/2018",  
            "address": "...",  
            "total": "9.00"  
        }  
          
        Returns dict mapping field -> list of dicts with keys:  
            - "ocr_text": raw OCR output  
            - "ground_truth": correct value for that field  
            - "filename": source filename  
        """  
        training_data_by_field = {field: [] for field in FIELDS}  
          
        # Get all OCR files  
        ocr_files = sorted([  
            f for f in os.listdir(self.ocr_dir)  
            if f.endswith(".txt")  
        ])  
          
        if sample_size:  
            ocr_files = ocr_files[:sample_size]  
          
        logger.info(f"Loading training data from {len(ocr_files)} files")  
          
        for fname in ocr_files:  
            ocr_path = os.path.join(self.ocr_dir, fname)  
            gt_path = os.path.join(self.gt_dir, fname)  # Same filename  
              
            # Load OCR text  
            try:  
                with open(ocr_path, 'r', encoding='utf-8') as f:  
                    ocr_text = f.read().strip()  
                    if not ocr_text:  
                        logger.warning(f"Empty OCR file: {fname}")  
                        continue  
            except Exception as e:  
                logger.error(f"Error reading OCR file {fname}: {e}")  
                continue  
              
            # Load ground truth  
            if not os.path.exists(gt_path):  
                logger.warning(f"Missing ground truth for {fname}")  
                continue  
              
            try:  
                with open(gt_path, 'r', encoding='utf-8') as f:  
                    content = f.read().strip()  
                    if not content:  
                        logger.warning(f"Empty ground truth file: {fname}")  
                        continue  
                      
                    gt_json = json.loads(content)  
                  
                if isinstance(gt_json, dict):  
                    for field in FIELDS:  
                        if field in gt_json:  
                            gt_value = gt_json[field]  
                              
                            # Only add if ground truth is not None/empty  
                            if gt_value and str(gt_value).strip():  
                                training_data_by_field[field].append({  
                                    "ocr_text": ocr_text,  
                                    "ground_truth": str(gt_value).strip(),  
                                    "filename": fname  
                                })  
                            else:  
                                logger.debug(f"Empty ground truth for field {field} in {fname}")  
              
            except json.JSONDecodeError as e:  
                logger.error(f"Invalid JSON in ground truth {fname}: {e}")  
                continue  
            except Exception as e:  
                logger.error(f"Error loading ground truth {fname}: {e}")  
                continue  
          
        logger.info("=" * 70)  
        logger.info("Training data loaded:")  
        for field, data in training_data_by_field.items():  
            logger.info(f"  {field}: {len(data)} examples")  
        logger.info("=" * 70)  
          
        return training_data_by_field  
      
    def train_ensemble_models(self,   
                             training_data_by_field: Dict[str, List[Dict]]  
                             ) -> Dict[str, EnsembleRegexExtractor]:  
        """  
        Train ensemble models for each field.  
          
        Returns: dict mapping field_name -> trained EnsembleRegexExtractor  
        """  
          
        logger.info("\n" + "=" * 70)  
        logger.info("TRAINING ENSEMBLE MODELS")  
        logger.info("=" * 70)  
          
        models = {}  
          
        for field in FIELDS:  
            logger.info(f"\nTraining ensemble for: {field.upper()}")  
            logger.info("-" * 70)  
              
            if not training_data_by_field[field]:  
                logger.warning(f"❌ No training data for field: {field}")  
                continue  
              
            # Create and train extractor  
            extractor = EnsembleRegexExtractor(  
                field_name=field,  
                n_patterns=config.ENSEMBLE_N_PATTERNS,  
                min_accuracy=config.ENSEMBLE_MIN_ACCURACY  
            )  
              
            result = extractor.learn_ensemble(  
                training_data_by_field[field],  
                variance_threshold=config.ENSEMBLE_VARIANCE_THRESHOLD  
            )  
              
            if result["success"]:  
                logger.info(f"✅ Training successful")  
                logger.info(f"   Candidates generated: {result['total_candidates']}")  
                logger.info(f"   Patterns selected: {result['n_patterns']}")  
                logger.info(f"   Average accuracy: {result['average_accuracy']:.3f}")  
                  
                logger.info("\n   Pattern details:")  
                for i, p in enumerate(result["patterns"], 1):  
                    pattern_display = p['pattern'][:60] + ("..." if len(p['pattern']) > 60 else "")  
                    logger.info(  
                        f"     {i}. {pattern_display}"  
                    )  
                    logger.info(  
                        f"        Weight={p['weight']}, Confidence={p['confidence']}"  
                    )  
                  
                # Save model  
                model_dir = config.ENSEMBLE_MODELS_DIR  
                os.makedirs(model_dir, exist_ok=True)  
                model_path = os.path.join(model_dir, f"{field}_ensemble_model.json")  
                extractor.save_model(model_path)  
                logger.info(f"   Model saved to: {model_path}")  
                  
                models[field] = extractor  
            else:  
                logger.error(f"❌ Training failed: {result.get('error')}")  
          
        return models  
      
    def evaluate_models(self,  
                       models: Dict[str, EnsembleRegexExtractor],  
                       test_ocr_dir: str = None) -> Dict:  
        """  
        Evaluate trained ensemble models on test data.  
          
        Compare:  
        - Baseline (hand-crafted regex)  
        - Ensemble (trained patterns)  
        """  
          
        if test_ocr_dir is None:  
            test_ocr_dir = self.ocr_dir  
          
        logger.info("\n" + "=" * 70)  
        logger.info("EVALUATING MODELS")  
        logger.info("=" * 70)  
          
        # Get test files  
        test_files = sorted([  
            f for f in os.listdir(test_ocr_dir)  
            if f.endswith(".txt")  
        ])  
          
        logger.info(f"Test files: {len(test_files)}")  
          
        results = {  
            "baseline": self._evaluate_baseline(test_files, test_ocr_dir),  
            "ensemble": self._evaluate_ensemble(models, test_files, test_ocr_dir)  
        }  
          
        # Compute improvements  
        results["improvement"] = self._compute_improvements(results)  
          
        return results  
      
    def _evaluate_baseline(self, test_files: List[str], test_ocr_dir: str) -> Dict:  
        """Evaluate hand-crafted baseline"""  
          
        logger.info("\n" + "-" * 70)  
        logger.info("BASELINE: Hand-crafted Regex (Tesseract)")  
        logger.info("-" * 70)  
          
        field_results = {field: [] for field in FIELDS}  
          
        for fname in test_files:  
            ocr_path = os.path.join(test_ocr_dir, fname)  
            gt_path = os.path.join(self.gt_dir, fname)  
              
            # Load OCR  
            try:  
                with open(ocr_path, 'r', encoding='utf-8') as f:  
                    ocr_text = f.read()  
            except Exception as e:  
                logger.warning(f"Error reading OCR {fname}: {e}")  
                continue  
              
            # Load ground truth  
            if not os.path.exists(gt_path):  
                continue  
              
            try:  
                with open(gt_path, 'r', encoding='utf-8') as f:  
                    gt_data = json.load(f)  
            except json.JSONDecodeError:  
                logger.warning(f"Invalid JSON in GT {fname}")  
                continue  
              
            # Extract with baseline  
            baseline_data = extract_tesseract_baseline(ocr_text)  
              
            # Compare fields  
            for field in FIELDS:  
                pred = baseline_data.get(field)  
                gt = gt_data.get(field)  
                  
                similarity = self._string_similarity(pred, gt)  
                field_results[field].append({  
                    "predicted": pred,  
                    "ground_truth": gt,  
                    "similarity": similarity,  
                    "filename": fname  
                })  
          
        # Compute aggregated metrics  
        metrics = self._compute_metrics(field_results)  
          
        logger.info("\nBaseline Accuracy per Field:")  
        for field in FIELDS:  
            acc = metrics[field]["accuracy"]  
            n = metrics[field]["n_samples"]  
            logger.info(f"  {field:<12}: {acc:.3f} ({n} samples)")  
          
        return {  
            "field_results": field_results,  
            "metrics": metrics  
        }  
      
    def _evaluate_ensemble(self,  
                          models: Dict[str, EnsembleRegexExtractor],  
                          test_files: List[str],  
                          test_ocr_dir: str) -> Dict:  
        """Evaluate trained ensemble"""  
          
        logger.info("\n" + "-" * 70)  
        logger.info("ENSEMBLE: Learned Patterns")  
        logger.info("-" * 70)  
          
        field_results = {field: [] for field in FIELDS}  
          
        for fname in test_files:  
            ocr_path = os.path.join(test_ocr_dir, fname)  
            gt_path = os.path.join(self.gt_dir, fname)  
              
            # Load OCR  
            try:  
                with open(ocr_path, 'r', encoding='utf-8') as f:  
                    ocr_text = f.read()  
            except Exception as e:  
                logger.warning(f"Error reading OCR {fname}: {e}")  
                continue  
              
            # Load ground truth  
            if not os.path.exists(gt_path):  
                continue  
              
            try:  
                with open(gt_path, 'r', encoding='utf-8') as f:  
                    gt_data = json.load(f)  
            except json.JSONDecodeError:  
                logger.warning(f"Invalid JSON in GT {fname}")  
                continue  
              
            # Extract with ensemble for each field  
            for field in FIELDS:  
                if field not in models:  
                    logger.debug(f"No model for field: {field}")  
                    continue  
                  
                model = models[field]  
                result = model.extract_with_confidence(ocr_text)  
                pred = result["value"]  
                confidence = result["confidence"]  
                agreement = result["agreement"]  
                  
                gt = gt_data.get(field)  
                similarity = self._string_similarity(pred, gt)  
                  
                field_results[field].append({  
                    "predicted": pred,  
                    "ground_truth": gt,  
                    "similarity": similarity,  
                    "confidence": confidence,  
                    "agreement": agreement,  
                    "filename": fname  
                })  
          
        # Compute aggregated metrics  
        metrics = self._compute_metrics(field_results)  
          
        logger.info("\nEnsemble Accuracy per Field:")  
        for field in FIELDS:  
            acc = metrics[field]["accuracy"]  
            n = metrics[field]["n_samples"]  
            logger.info(f"  {field:<12}: {acc:.3f} ({n} samples)")  
          
        return {  
            "field_results": field_results,  
            "metrics": metrics  
        }  
      
    def _compute_improvements(self, results: Dict) -> Dict:  
        """Compute improvement from baseline to ensemble"""  
        improvements = {}  
          
        for field in FIELDS:  
            baseline_acc = results["baseline"]["metrics"][field]["accuracy"]  
            ensemble_acc = results["ensemble"]["metrics"][field]["accuracy"]  
            improvement = ensemble_acc - baseline_acc  
            improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0  
              
            improvements[field] = {  
                "absolute_improvement": round(improvement, 4),  
                "percentage_improvement": round(improvement_pct, 2)  
            }  
          
        return improvements  
      
    @staticmethod  
    def _string_similarity(s1: str, s2: str) -> float:  
        """Compute string similarity 0-1"""  
        if not s1 or not s2:  
            # Both None/empty = perfect match  
            # One None/empty = no match  
            return 1.0 if (not s1 and not s2) else 0.0  
          
        s1_clean = str(s1).lower().strip()  
        s2_clean = str(s2).lower().strip()  
        return SequenceMatcher(None, s1_clean, s2_clean).ratio()  
      
    @staticmethod  
    def _compute_metrics(field_results: Dict[str, List[Dict]]) -> Dict:  
        """Compute accuracy per field"""  
        metrics = {}  
          
        for field, results in field_results.items():  
            if not results:  
                metrics[field] = {  
                    "accuracy": 0.0,  
                    "n_samples": 0,  
                    "correct": 0  
                }  
                continue  
              
            accuracies = [r["similarity"] for r in results]  
            correct = sum(1 for acc in accuracies if acc >= 0.8)  # Threshold  
              
            metrics[field] = {  
                "accuracy": round(sum(accuracies) / len(accuracies), 4),  
                "n_samples": len(results),  
                "correct": correct,  
                "correct_percentage": round(correct / len(results) * 100, 2) if results else 0  
            }  
          
        return metrics  
      
    def print_summary(self, results: Dict) -> None:  
        """Print evaluation summary"""  
          
        logger.info("\n" + "=" * 70)  
        logger.info("EVALUATION SUMMARY")  
        logger.info("=" * 70)  
          
        for field in FIELDS:  
            baseline_acc = results["baseline"]["metrics"][field]["accuracy"]  
            ensemble_acc = results["ensemble"]["metrics"][field]["accuracy"]  
            improvement = results["improvement"][field]["absolute_improvement"]  
            improvement_pct = results["improvement"][field]["percentage_improvement"]  
              
            logger.info(f"\n{field.upper()}")  
            logger.info(f"  Baseline:  {baseline_acc:.4f}")  
            logger.info(f"  Ensemble:  {ensemble_acc:.4f}")  
            logger.info(f"  Change:    {improvement:+.4f} ({improvement_pct:+.2f}%)")  
              
            if improvement > 0:  
                logger.info(f"  ✅ IMPROVED")  
            elif improvement < 0:  
                logger.info(f"  ❌ DECLINED")  
            else:  
                logger.info(f"  ➡️  NO CHANGE")  
      
    def save_results(self, results: Dict, output_path: str = "ensemble_evaluation_results.json"):  
        """Save evaluation results to JSON"""  
          
        # Convert numpy types to native Python types for JSON serialization  
        def convert_types(obj):  
            if isinstance(obj, np.floating):  
                return float(obj)  
            elif isinstance(obj, np.integer):  
                return int(obj)  
            elif isinstance(obj, dict):  
                return {k: convert_types(v) for k, v in obj.items()}  
            elif isinstance(obj, list):  
                return [convert_types(item) for item in obj]  
            return obj  
          
        results_serializable = convert_types(results)  
          
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)  
          
        with open(output_path, 'w') as f:  
            json.dump(results_serializable, f, indent=2)  
          
        logger.info(f"\n✅ Results saved to {output_path}")  
      
    def log_to_mlflow(self, results: Dict) -> None:  
        """Log evaluation results to MLflow"""  
          
        mlflow.set_experiment("Ensemble Regex Extractor")  
          
        with mlflow.start_run(run_name="Ensemble_vs_Baseline"):  
              
            # Log baseline metrics  
            baseline_metrics = results["baseline"]["metrics"]  
            for field in FIELDS:  
                accuracy = baseline_metrics[field]["accuracy"]  
                mlflow.log_metric(f"baseline_{field}_accuracy", accuracy)  
              
            # Log ensemble metrics  
            ensemble_metrics = results["ensemble"]["metrics"]  
            for field in FIELDS:  
                accuracy = ensemble_metrics[field]["accuracy"]  
                mlflow.log_metric(f"ensemble_{field}_accuracy", accuracy)  
              
            # Log improvements  
            for field in FIELDS:  
                improvement = results["improvement"][field]["absolute_improvement"]  
                improvement_pct = results["improvement"][field]["percentage_improvement"]  
                mlflow.log_metric(f"improvement_{field}", improvement)  
                mlflow.log_metric(f"improvement_pct_{field}", improvement_pct)  
              
            # Log parameters  
            mlflow.log_param("n_patterns", config.ENSEMBLE_N_PATTERNS)  
            mlflow.log_param("variance_threshold", config.ENSEMBLE_VARIANCE_THRESHOLD)  
            mlflow.log_param("min_accuracy", config.ENSEMBLE_MIN_ACCURACY)  
          
        logger.info("✅ Results logged to MLflow")  
  
  
def main():  
    """Run complete evaluation pipeline"""  
      
    logger.info("\n" + "=" * 70)  
    logger.info("ENSEMBLE REGEX EXTRACTOR - COMPLETE PIPELINE")  
    logger.info("=" * 70 + "\n")  
      
    # Setup  
    ocr_dir = config.OUTPUT_DIR_PADDLE  # Use PaddleOCR output  
    gt_dir = config.GROUND_TRUTH_DIR  
      
    # Verify directories  
    if not os.path.exists(ocr_dir):  
        logger.error(f"OCR directory not found: {ocr_dir}")  
        return  
      
    if not os.path.exists(gt_dir):  
        logger.error(f"Ground truth directory not found: {gt_dir}")  
        return  
      
    evaluator = EnsembleEvaluator(ocr_dir, gt_dir)  
      
    # STAGE 1: Load training data  
    logger.info("\nSTAGE 1: Loading Training Data")  
    logger.info("-" * 70)  
    training_data = evaluator.load_training_data(sample_size=None)  # Use all available  
      
    # Check if we have data  
    if not any(training_data.values()):  
        logger.error("No training data found!")  
        return  
      
    # STAGE 2: Train ensemble models  
    logger.info("\nSTAGE 2: Training Ensemble Models")  
    logger.info("-" * 70)  
    models = evaluator.train_ensemble_models(training_data)  
      
    if not models:  
        logger.error("Failed to train any models!")  
        return  
      
    # STAGE 3: Evaluate on same data  
    logger.info("\nSTAGE 3: Evaluating Models")  
    logger.info("-" * 70)  
    results = evaluator.evaluate_models(models)  
      
    # STAGE 4: Print summary  
    evaluator.print_summary(results)  
      
    # STAGE 5: Save results  
    evaluator.save_results(results)  
      
    # STAGE 6: Log to MLflow  
    try:  
        evaluator.log_to_mlflow(results)  
    except Exception as e:  
        logger.warning(f"⚠️ Could not log to MLflow: {e}")  
      
    logger.info("\n" + "=" * 70)  
    logger.info("PIPELINE COMPLETE")  
    logger.info("=" * 70 + "\n")  
  
  
if __name__ == "__main__":  
    main()  