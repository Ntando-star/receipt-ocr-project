# src/standardize_test_results.py
import json
from pathlib import Path

def standardize_results(ocr_result_file: str, output_file: str, test_files_dir: str):
    """
    Standardize OCR results JSON to include only test files and proper structure
    compatible with ComparisonAnalyzer.
    """
    test_files = sorted([f.name for f in Path(test_files_dir).glob("*.txt")])

    with open(ocr_result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    standardized = {"test_results": {"baseline": {}, "ensemble": {}}}

    for method in ["baseline", "ensemble"]:
        standardized["test_results"][method] = {}
        for field in ["company", "address", "date", "total"]:
            field_results = data.get("test_results", {}).get(method, {}).get(field, [])
            # Filter only files that exist in test set
            filtered = [r for r in field_results if r.get("filename") in test_files]
            standardized["test_results"][method][field] = filtered

    # Save standardized JSON
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(standardized, f, indent=2)

    print(f"✅ Standardized results saved to {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python standardize_test_results.py <ocr_result.json> <output.json> <test_files_dir>")
        sys.exit(1)

    ocr_result_file = sys.argv[1]
    output_file = sys.argv[2]
    test_files_dir = sys.argv[3]

    standardize_results(ocr_result_file, output_file, test_files_dir)
