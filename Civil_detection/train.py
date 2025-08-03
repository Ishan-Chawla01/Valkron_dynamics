from ultralytics import YOLO
import os
import torch
import multiprocessing

torch.cuda.empty_cache()


DATA_YAML_PATH = os.path.abspath('visdrone_yolo_dataset_det/visdrone.yaml') 

MODEL_NAME = 'yolov8s.pt'

# Training parameters
EPOCHS = 20 # Increased epochs for better convergence on complex datasets(changed to 20 for local)
IMG_SIZE = 640 # Increased image size to better preserve small object details
BATCH_SIZE = 4 # Increased batch size for more stable training (adjust based on GPU memory)
DEVICE = 0 # Assuming GPU 0 is available. Set to 'cpu' if no GPU, or multiple GPUs '0,1'
NAME_RUN = 'visdrone_aerial_detector_v2' # New name for the training run
WORKERS = 4 


if __name__ == '__main__':
    
    model = YOLO(MODEL_NAME)
    print(f"\n--- Starting YOLOv8 Training ---")
    print(f"Dataset: {DATA_YAML_PATH}") # This will now print the full absolute path
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Run Name: {NAME_RUN}")
    print(f"Workers: {WORKERS}")
    print("-" * 30)

    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name=NAME_RUN,
        workers=WORKERS, 
        )

    print(f"\n--- Training Complete ---")
    print(f"Results saved to: runs/detect/{NAME_RUN}/")
