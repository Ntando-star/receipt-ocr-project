import cv2
import numpy as np
from pathlib import Path
import os

def preprocess_image(img_path):
    """Load image as grayscale and apply adaptive threshold."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    return img

# ADD THIS FUNCTION TO src/preprocessing.py  
  
def segment_lines_with_positions(img):  
    """  
    Segment image into lines AND track their Y-positions.  
    Returns list of tuples: (line_image, y_position)  
    """  
    # Get the lines using your existing function  
    lines = segment_lines(img)  
      
    # Calculate Y position for each line  
    # We need to estimate where each line appears vertically  
    img_height = img.shape[0]  
    num_lines = len(lines)  
      
    # Simple approach: divide image height by number of lines  
    line_height_estimate = img_height // num_lines if num_lines > 0 else 1  
      
    # Create tuples of (line, y_position)  
    lines_with_pos = []  
    for i, line in enumerate(lines):  
        y_pos = i * line_height_estimate  
        lines_with_pos.append((line, y_pos))  
      
    return lines_with_pos  

def segment_lines(img):
    """Segment image into lines using morphology and contours (in-memory)."""
    h, w = img.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 40, 3))
    dilated = cv2.dilate(img, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for c in sorted(contours, key=lambda x: cv2.boundingRect(x)[1]):
        x, y, w_box, h_box = cv2.boundingRect(c)
        if h_box > 10 and w_box > 20:  # filter tiny noise
            line_img = img[y:y+h_box, x:x+w_box]
            lines.append(line_img)
    return lines

def resize_and_pad(line_img, target_height=384, target_width=384):
    """Resize line proportionally and pad to target size."""
    h, w = line_img.shape
    new_h = target_height
    new_w = int(w * (target_height / h))
    if new_w > target_width:
        new_w = target_width
    resized = cv2.resize(line_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
    padded[:, :new_w] = resized
    return padded

def preprocess_file(img_path, output_dir="data/preprocessed"):
    """
    Full preprocessing pipeline:
      1. Preprocess image
      2. Segment into lines
      3. Resize and pad
      4. Save preprocessed lines to output_dir

    Returns:
        List of file paths to preprocessed line images
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    img = preprocess_image(img_path)
    lines = segment_lines(img)

    line_paths = []
    base_name = Path(img_path).stem
    for idx, line_img in enumerate(lines):
        padded = resize_and_pad(line_img)
        line_path = os.path.join(output_dir, f"{base_name}_line{idx}.png")
        cv2.imwrite(line_path, padded)
        line_paths.append(line_path)

    return line_paths
