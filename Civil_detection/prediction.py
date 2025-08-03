from ultralytics import YOLO

model_path = r'runs/detect/visdrone_aerial_detector_v2/weights/best.pt' 
model = YOLO(model_path)

video_path = r'D:/all_python_projects/UAV/test.mp4' 


CIVILIAN_VEHICLE_CLASS_IDS = [0, 1, 2, 3, 4] 

results = model.predict(
    source=video_path, 
    save=True,           
    conf=0.30,           
    iou=0.45,            
    classes=CIVILIAN_VEHICLE_CLASS_IDS 
)
