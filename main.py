# imports
from ultralytics import YOLO

# config
model = YOLO("hizo9coco.pt")
results = model(1, show=True)

# code
for result in results:
    boxes = result.boxes
    classes = result.names