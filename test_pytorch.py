import torch
import SimpleNet 
from ultralytics import YOLO



# Carica un modello pre-allenato di YOLO (ad es. YOLOv8)
model = YOLO('yolov8n.pt')
print(model)