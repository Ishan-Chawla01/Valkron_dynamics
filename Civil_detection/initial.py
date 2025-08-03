import os
import cv2
import numpy as np
from tqdm import tqdm 
from sklearn.model_selection import train_test_split 

# --- Configuration ---
VISDRONE_DATASET_ROOT = 'VisDrone2019-DET-train' 
OUTPUT_ROOT = 'visdrone_yolo_dataset_det' 
TARGET_CLASSES = {
    
    4: 0,   # Car -> 0
    9: 1,   # Bus -> 1
    6: 2,   # Truck -> 2
    3: 3,   # Bicycle -> 3
    10: 4,  # Motor -> 4
    1: 5,   # Pedestrian -> 5
    2: 6    # People -> 6
}

YOLO_CLASS_NAMES = [
    "car", "bus", "truck", "bicycle", "motor", "pedestrian", "people"
]

# Train/Validation split ratio for individual images
TRAIN_RATIO = 0.8 

# --- Create Output Directories ---
os.makedirs(os.path.join(OUTPUT_ROOT, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', 'val'), exist_ok=True)

# Path to VisDrone images and annotations for DET
images_dir = os.path.join(VISDRONE_DATASET_ROOT, 'images')
annotations_dir = os.path.join(VISDRONE_DATASET_ROOT, 'annotations')

# Get all image files
all_image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
all_image_files.sort() # Ensure consistent order

# Split images directly into train and validation sets
train_image_files, val_image_files = train_test_split(all_image_files, test_size=(1 - TRAIN_RATIO), random_state=42)

print(f"Total images: {len(all_image_files)}")
print(f"Train images: {len(train_image_files)}")
print(f"Validation images: {len(val_image_files)}")


# --- Processing Function ---
def process_images_for_split(image_list, split_type):
    print(f"Processing {len(image_list)} images for {split_type} split...")
    for img_filename in tqdm(image_list, desc=f"Processing {split_type} images"):
        img_path = os.path.join(images_dir, img_filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        h, w, _ = img.shape

        # Annotation file for DET dataset has the same name as the image but with .txt extension
        anno_filename = os.path.splitext(img_filename)[0] + '.txt'
        anno_file_path = os.path.join(annotations_dir, anno_filename) 

        # Construct YOLO path
        output_img_path = os.path.join(OUTPUT_ROOT, 'images', split_type, img_filename)
        output_label_path = os.path.join(OUTPUT_ROOT, 'labels', split_type, anno_filename)

        # Save image
        cv2.imwrite(output_img_path, img)

        # Write YOLO annotation file
        with open(output_label_path, 'w') as f_label:
            if os.path.exists(anno_file_path): # Only read if annotation file exists
                with open(anno_file_path, 'r') as f_anno:
                    for line in f_anno:
                        line = line.strip() # Remove leading/trailing whitespace
                        if not line: # Skip empty lines
                            continue
                        
                        try:
                            str_parts = [p for p in line.split(',') if p]
                            parts = list(map(int, str_parts))
                            
                            if len(parts) != 8:
                                print(f"Warning: Skipping malformed annotation line in {anno_file_path}: '{line}'. Expected 8 values, got {len(parts)}.")
                                continue

                            bbox_left, bbox_top, bbox_width, bbox_height, score, obj_category, truncation, occlusion = parts

                            if obj_category not in TARGET_CLASSES:
                                continue

                            yolo_class_id = TARGET_CLASSES[obj_category]

                            center_x = (bbox_left + bbox_width / 2) / w
                            center_y = (bbox_top + bbox_height / 2) / h
                            norm_width = bbox_width / w
                            norm_height = bbox_height / h

                            center_x = max(0.0, min(1.0, center_x))
                            center_y = max(0.0, min(1.0, center_y))
                            norm_width = max(0.0, min(1.0, norm_width))
                            norm_height = max(0.0, min(1.0, norm_height))

                            f_label.write(f"{yolo_class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                        
                        except ValueError as e:
                            print(f"Warning: Skipping invalid annotation line in {anno_file_path}: '{line}'. Error: {e}")
                            continue
                        except IndexError as e:
                            print(f"Warning: Skipping incomplete annotation line in {anno_file_path}: '{line}'. Error: {e}. Not enough parts to unpack.")
                            continue
                    
# --- Run Processing ---
process_images_for_split(train_image_files, 'train')
process_images_for_split(val_image_files, 'val')

print("Dataset preparation complete!")
print(f"YOLO formatted dataset saved to: {OUTPUT_ROOT}")


yaml_content = f"""
path: {os.path.abspath(OUTPUT_ROOT)}
train: images/train
val: images/val

nc: {len(YOLO_CLASS_NAMES)}

names: {YOLO_CLASS_NAMES}
"""

with open(os.path.join(OUTPUT_ROOT, 'visdrone.yaml'), 'w') as f:
    f.write(yaml_content)

print(f"\nGenerated dataset.yaml file: {os.path.join(OUTPUT_ROOT, 'visdrone.yaml')}")
