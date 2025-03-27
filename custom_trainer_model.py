from functools import partial
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from data_loader import YOLODataset , train_transforms
from  Custom_Dataloader import Custom_Dataloader, custom_transforms
from custom_trainer import CustomTrainer
import os
import wandb
import glob





img_dir = '../dataset/train/images'
label_dir = '../dataset/train/labels'

print("🔍 Debug: Verifica contenuto cartella immagini")
for ext in ('*.jpg', '*.jpeg', '*.png'):
    files = glob.glob(os.path.join(img_dir, ext))
    print(f"Estensione {ext}: trovate {len(files)} immagini")

train_dataset = Custom_Dataloader(img_dir, label_dir, transform=custom_transforms)

print(f"Found {len(train_dataset)} images for training")
print(f"Example image path: {train_dataset.img_paths[0] if len(train_dataset) > 0 else 'None'}")

test_name ="yolov9s_custom"
folder = "testing"

TrainerWithDataset = partial(CustomTrainer, custom_train_dataset=train_dataset)

wandb.init(project="YOLO_PROJECT", name=test_name, resume="allow")
 

model = YOLO("yolov9s.pt")
results = model.train(
    data="dataset.yaml",
    device=0,
    epochs=200,
    imgsz=640,      
    name=test_name,
    project = folder,
    trainer=TrainerWithDataset,
    plots = True,
    resume = False
    
)

train_dataset.report_stats()

print(test_name + " - Training complete ")
print("All  result saved in " + folder )

best_weights = os.path.join(folder, "weights", "best.pt")
print(f"Best weights  for "  +  test_name + " saved at: {best_weights}")

validation_name = test_name + "_validation"
metrics = model.val(
    name=validation_name,
    project = folder,
)

print(validation_name + " - Validation complete ")
print("All  result saved in " + folder + "/validation_name ")