from ultralytics import YOLO, settings
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch



# Carica il modello (qui usiamo il modello YOLOv8n pre-addestrato)
model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")
model = YOLO("yolov8n.yaml").load("yolov8n.pt")

model.info()


results = model.train(
    data="dataset.yaml",       
    epochs=100,                
    imgsz=640,                 
    device=0,
    optimizer='AdamW',
    auto_augment='randaugment',
    resume = True,
    name = "test_1",
    plots = True
)

metrics = model.val()

metrics.top1  # top1 accuracy
metrics.top5  # top5 accuracy

# YOLOv8 salva i risultati del training in una directory (di solito runs/train/exp*)
# Se l'oggetto results possiede l'attributo "path", lo usiamo per individuare la directory di log:

train_dir = results.path if hasattr(results, "path") else "runs/train/exp"

print(f"Training completato. I risultati sono salvati in: {train_dir}")
