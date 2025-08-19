from ultralytics import YOLO

# Load a pre-trained YOLOv8 model to start from
# Using yolov8s.pt (small) is a good balance of speed and accuracy
model = YOLO('yolov8s.pt')

# Train the model on your custom dataset
if __name__ == '__main__':
    results = model.train(
        data=r'C:\Users\Lenovo\Desktop\work_d\laptop_config.yaml',  # Path to your dataset config file
        epochs=10,               # Number of training rounds (50 is a good start)
        imgsz=640,                  # Image size for training
        name='laptop_detector_v1'   # A name for this training run
    )