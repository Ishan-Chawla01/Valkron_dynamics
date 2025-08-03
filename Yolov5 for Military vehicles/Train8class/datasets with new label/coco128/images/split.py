import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Image and label directories
image_dir = r"C:\MY files\Valkan dynamics\ArmaCV_dataset\resized_images"
label_dir = r"C:\MY files\Valkan dynamics\ArmaCV_dataset\labelsMod"

# Get all image files
all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Train-validation split ratio
train_ratio = 0.5

# Shuffle and split
random.shuffle(all_images)
split_idx = int(len(all_images) * train_ratio)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

image_dir_o = r"C:\MY files\Valkan dynamics\Test_ArmaDataset\datasets with new label\coco128\images"
label_dir_o = r"C:\MY files\Valkan dynamics\Test_ArmaDataset\datasets with new label\coco128\labels"

# Output subdirectories
image_train_dir = os.path.join(image_dir_o, 'train2017')
image_val_dir = os.path.join(image_dir_o, 'val2017')
label_train_dir = os.path.join(label_dir_o, 'train2017')
label_val_dir = os.path.join(label_dir_o, 'val2017')

# Create output folders
os.makedirs(image_train_dir, exist_ok=True)
os.makedirs(image_val_dir, exist_ok=True)
os.makedirs(label_train_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

def copy_pair(image_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for img_file in image_list:
        # Copy image
        shutil.copy(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))

        # Copy corresponding label (.txt)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_src_path = os.path.join(src_lbl_dir, label_file)
        label_dst_path = os.path.join(dst_lbl_dir, label_file)

        # Only copy label if it exists
        if os.path.exists(label_src_path):
            shutil.copy(label_src_path, label_dst_path)

# Copy files to train and val folders
copy_pair(train_images, image_dir, label_dir, image_train_dir, label_train_dir)
copy_pair(val_images, image_dir, label_dir, image_val_dir, label_val_dir)

print(f"Split complete: {len(train_images)} training and {len(val_images)} validation images (with labels).")
