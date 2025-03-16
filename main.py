# imports
from ultralytics import YOLO

# config
model = YOLO("models/yolo11n.pt")
results = model(1, show=True, conf=0.5)

# code
for result in results:
    boxes = result.boxes
    classes = result.names
