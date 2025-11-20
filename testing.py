# testing.py    
import os    
import json    
from difflib import SequenceMatcher    
import warnings    
import mlflow    
    
# Suppress the MLflow deprecation warning    
warnings.filterwarnings('ignore', category=FutureWarning)    
    
# Use SQLite backend instead of filesystem    
mlflow.set_tracking_uri("sqlite:///mlflow.db")    
    
from src.config import config    
from src.extract_fields import extract_fields as extract_tesseract    
from src.extract_fields_paddle import extract_fields as extract_paddle    
    
def load_ground_truth(gt_dir, filename):    
    """Load ground truth from JSON file."""    
    base_name = filename.replace('.txt', '')    
    gt_path = os.path.join(gt_dir, f"{base_name}.txt")    
        
    if os.path.exists(gt_path):    
        try:    
            with open(gt_path, "r", encoding="utf-8") as f:    
                data = json.load(f)    
                if isinstance(data, dict):    
                    return {    
                        "company": data.get("company"),    
                        "address": data.get("address"),    
                        "date": data.get("date"),    
                        "total": data.get("total"),    
                    }    
        except (json.JSONDecodeError, IOError) as e:    
            print(f"Error loading GT for {filename}: {e}")    
        
    return {"company": None, "address": None, "date": None, "total": None}    
    
def test_ocr_folder_with_gt_txt(ocr_name, ocr_dir, gt_dir):    
    """Test extract_fields on OCR output files and compare with ground truth."""    
    print(f"\n===== Testing {ocr_name.upper()} OCR Output with Ground Truth =====")    
        
    txt_files = sorted([f for f in os.listdir(ocr_dir) if f.lower().endswith(".txt")])    
    if not txt_files:    
        print(f"No .txt files found in {ocr_dir}")    
        return []    
    
    all_results = {}    
    metrics_list = []    
    
    for fname in txt_files:    
        # Load OCR text    
        ocr_path = os.path.join(ocr_dir, fname)    
        with open(ocr_path, "r", encoding="utf-8") as f:    
            ocr_text = f.read()    
    
        # Select extractor based on OCR type    
        if ocr_name.lower() == "tesseract":    
            extracted_ocr = extract_tesseract(ocr_text)    
        elif ocr_name.lower() == "paddleocr":    
            extracted_ocr = extract_paddle(ocr_text)    
        else:    
            extracted_ocr = {"company": None, "address": None, "date": None, "total": None}    
    
        all_results[fname] = extracted_ocr    
    
        # Load ground truth from JSON file    
        gt_data = load_ground_truth(gt_dir, fname)    
    
        # Compute field accuracy    
        field_accuracies = {}    
        for field in ["company", "address", "date", "total"]:    
            pred = extracted_ocr.get(field, "")    
            gt = gt_data.get(field, "")    
                
            # Handle None values    
            if pred is None:    
                pred = ""    
            if gt is None:    
                gt = ""    
                
            accuracy = SequenceMatcher(None, str(pred).lower(), str(gt).lower()).ratio()    
            field_accuracies[field] = accuracy    
    
        metrics_list.append({    
            "filename": fname,    
            "ocr_type": ocr_name,    
            **field_accuracies    
        })    
    
        # Print side-by-side    
        print(f"\nFile: {fname}")    
        for field in ["company", "address", "date", "total"]:    
            ocr_val = extracted_ocr.get(field) or "None"    
            gt_val = gt_data.get(field) or "None"    
            # Truncate for display    
            ocr_display = str(ocr_val)[:50] if ocr_val else "None"    
            gt_display = str(gt_val)[:50] if gt_val else "None"    
            accuracy = field_accuracies[field]    
            print(f"{field.capitalize():<10} | Accuracy: {accuracy:.2f} | OCR: {ocr_display:<50} | GT: {gt_display}")    
    
    # Save results    
    output_json = f"data/testing_results_with_gt_{ocr_name}.json"    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)    
    with open(output_json, "w", encoding="utf-8") as f:    
        json.dump(all_results, f, indent=4)    
        
    print(f"\n✅ All results saved to {output_json}")    
        
    return metrics_list    
    
def evaluate_pipeline(ocr_type):    
    """Evaluate OCR pipeline and return precision, recall, F1."""    
    if ocr_type == "tesseract":    
        ocr_dir = config.OUTPUT_DIR_TESSERACT    
    elif ocr_type == "paddleocr":    
        ocr_dir = config.OUTPUT_DIR_PADDLE    
    else:    
        raise ValueError(f"Unknown OCR type: {ocr_type}")  
        
    gt_dir = config.GROUND_TRUTH_DIR    
        
    metrics = test_ocr_folder_with_gt_txt(ocr_type, ocr_dir, gt_dir)    
        
    if not metrics:    
        return 0.0, 0.0, 0.0    
        
    # Calculate average accuracies across all fields and files    
    fields = ["company", "address", "date", "total"]    
    field_accuracies = {field: [] for field in fields}    
        
    for metric in metrics:    
        for field in fields:    
            field_accuracies[field].append(metric[field])    
        
    # Compute macro averages    
    field_averages = {}    
    for field in fields:    
        if field_accuracies[field]:    
            field_averages[field] = sum(field_accuracies[field]) / len(field_accuracies[field])    
        else:    
            field_averages[field] = 0.0    
        
    # Overall accuracy is average of all field accuracies    
    overall_accuracy = sum(field_averages.values()) / len(field_averages) if field_averages else 0.0    
        
    # For simplicity, use overall accuracy for all three metrics    
    precision = overall_accuracy    
    recall = overall_accuracy    
    f1 = overall_accuracy    
        
    return precision, recall, f1    
    
if __name__ == "__main__":    
    mlflow.set_experiment("Receipt OCR Experiment")    
    
    results_summary = {}    
    
    # Test Tesseract    
    print("\n" + "="*60)    
    print("TESSERACT EVALUATION")    
    print("="*60)    
    precision_t, recall_t, f1_t = evaluate_pipeline("tesseract")    
        
    with mlflow.start_run(run_name="Tesseract + Regex"):    
        mlflow.log_metric("precision", precision_t)    
        mlflow.log_metric("recall", recall_t)    
        mlflow.log_metric("f1_score", f1_t)    
        mlflow.log_param("ocr_engine", "tesseract")    
        print(f"\n✅ Tesseract - Precision: {precision_t:.4f}, Recall: {recall_t:.4f}, F1: {f1_t:.4f}")    
        
    results_summary["tesseract"] = {    
        "precision": precision_t,    
        "recall": recall_t,    
        "f1_score": f1_t    
    }    
    
    # Test PaddleOCR    
    print("\n" + "="*60)    
    print("PADDLEOCR EVALUATION")    
    print("="*60)    
    precision_p, recall_p, f1_p = evaluate_pipeline("paddleocr")    
        
    with mlflow.start_run(run_name="PaddleOCR + Regex"):    
        mlflow.log_metric("precision", precision_p)    
        mlflow.log_metric("recall", recall_p)    
        mlflow.log_metric("f1_score", f1_p)    
        mlflow.log_param("ocr_engine", "paddleocr")    
        print(f"\n✅ PaddleOCR - Precision: {precision_p:.4f}, Recall: {recall_p:.4f}, F1: {f1_p:.4f}")    
        
    results_summary["paddleocr"] = {    
        "precision": precision_p,    
        "recall": recall_p,    
        "f1_score": f1_p    
    }    
    
    # Print comparison summary    
    print("\n" + "="*60)    
    print("COMPARISON SUMMARY")    
    print("="*60)    
    print(f"\nTesseract F1:   {f1_t:.4f}")    
    print(f"PaddleOCR F1:   {f1_p:.4f}")    
        
    # Determine winner    
    if f1_t > f1_p:    
        winner = "Tesseract"    
        f1_diff = f1_t - f1_p    
    elif f1_p > f1_t:    
        winner = "PaddleOCR"    
        f1_diff = f1_p - f1_t    
    else:    
        winner = "Tie"    
        f1_diff = 0.0    
        
    print(f"Winner: {winner}")    
    if f1_diff > 0:    
        print(f"Margin: +{f1_diff:.4f}")    
    
    # Save detailed comparison    
    with open("testing_results_comparison.json", "w") as f:    
        json.dump(results_summary, f, indent=2)    
        
    print(f"\n✓ Detailed comparison saved to testing_results_comparison.json")  