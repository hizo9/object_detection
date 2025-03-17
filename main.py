# imports
from ultralytics import YOLO

# configs
model = YOLO("yolo11m.pt")
results = model(1, show=True, conf=0.5)

# code
for result in results:
    boxes = result.boxes
    classes = result.names