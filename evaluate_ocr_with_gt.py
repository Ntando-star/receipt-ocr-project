# evaluate_ocr_with_gt.py    
import os    
import json    
from difflib import SequenceMatcher    
import mlflow    
from src.extract_fields import extract_fields  # For Tesseract    
from src.extract_fields_paddle import extract_fields as extract_fields_paddle  # For PaddleOCR    

# Directories    
pred_dir_tesseract = "data/ocr_output_tesseract_test"    
pred_dir_paddle = "data/ocr_output_paddle_test"    
gt_dir = "data/raw/ground_truth"    
    
FIELDS = ["company", "address", "date", "total"]    
    
def string_similarity(pred, gt):    
    """Compute a similarity score between two strings (0-1)."""    
    if not pred or not gt:    
        return 0.0    
    return SequenceMatcher(None, pred.strip().lower(), gt.strip().lower()).ratio()    
    
def evaluate(pred_dir, gt_dir, extractor):    
    """Evaluate OCR+Regex predictions against ground truth in-memory."""    
    tp, fp, fn = 0, 0, 0    
    results = []    
    
    for fname in os.listdir(pred_dir):    
        if not fname.endswith(".txt"):    
            continue    
    
        pred_path = os.path.join(pred_dir, fname)    
        gt_path = os.path.join(gt_dir, fname)    
    
        if not os.path.exists(gt_path):    
            print(f"⚠️ Missing ground truth for {fname}")    
            continue    
    
        # Load OCR text    
        with open(pred_path, "r", encoding="utf-8") as f:    
            ocr_text = f.read()    
    
        # Extract fields    
        pred_data = extractor(ocr_text)    
    
        # Load ground truth    
        with open(gt_path, "r", encoding="utf-8") as f:    
            try:    
                gt_data = json.loads(f.read())    
            except json.JSONDecodeError:    
                print(f"⚠️ Invalid ground truth JSON in {gt_path}")    
                continue    
    
        # Compare fields    
        file_results = {    
            "file": fname,    
            "fields": {}    
        }    
            
        for field in FIELDS:    
            pred_val = pred_data.get(field)    
            gt_val = gt_data.get(field)    
            similarity = string_similarity(pred_val, gt_val)    
    
            file_results["fields"][field] = {    
                "predicted": pred_val,    
                "ground_truth": gt_val,    
                "similarity": round(similarity, 3)    
            }    
    
            if pred_val and gt_val:    
                if similarity > 0.8:  # threshold    
                    tp += 1    
                else:    
                    fp += 1    
            elif pred_val and not gt_val:    
                fp += 1    
            elif not pred_val and gt_val:    
                fn += 1    
    
        results.append(file_results)    
    
    precision = tp / (tp + fp + 1e-6)    
    recall = tp / (tp + fn + 1e-6)    
    f1 = 2 * precision * recall / (precision + recall + 1e-6)    
        
    return round(precision, 3), round(recall, 3), round(f1, 3), results    
    
    
if __name__ == "__main__":    
    mlflow.set_experiment("Receipt OCR Comparison")    
    
    engines = [    
        ("tesseract", pred_dir_tesseract, extract_fields),    
        ("paddleocr", pred_dir_paddle, extract_fields_paddle)
    ]    
    
    all_results = {}    
    
    for engine_name, pred_dir, extractor in engines:    
        print(f"\n{'='*60}")    
        print(f"Evaluating {engine_name.upper()}")    
        print(f"{'='*60}")    
            
        precision, recall, f1, results = evaluate(pred_dir, gt_dir, extractor)    
    
        print(f"\n===== {engine_name.upper()} RESULTS =====")    
        print(f"Precision: {precision}")    
        print(f"Recall:    {recall}")    
        print(f"F1-score:  {f1}")    
    
        # Log to MLflow    
        with mlflow.start_run(run_name=f"{engine_name.upper()}_Run"):    
            mlflow.log_metric("precision", precision)    
            mlflow.log_metric("recall", recall)    
            mlflow.log_metric("f1_score", f1)    
            mlflow.log_param("ocr_engine", engine_name)    
    
        all_results[engine_name] = {    
            "precision": precision,    
            "recall": recall,    
            "f1_score": f1,    
            "file_results": results    
        }    
    
    # Print comparison    
    print(f"\n{'='*60}")    
    print("COMPARISON SUMMARY")    
    print(f"{'='*60}")    
        
    for engine_name, metrics in all_results.items():    
        print(f"\n{engine_name.upper()}:")    
        print(f"  Precision: {metrics['precision']}")    
        print(f"  Recall:    {metrics['recall']}")    
        print(f"  F1-score:  {metrics['f1_score']}")    
    
    # Save detailed results    
    with open("evaluation_results.json", "w") as f:    
        json.dump(all_results, f, indent=2)    
        
    print(f"\n✓ Detailed results saved to evaluation_results.json")  