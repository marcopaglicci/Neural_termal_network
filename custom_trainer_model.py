from functools import partial
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from data_loader import YOLODataset , train_transforms
#from  Custom_Dataloader import Custom_Dataloader, custom_transforms
from custom_trainer import CustomTrainer
from MultiTrasform_Dataloader import Custom_Dataloader, transforms_list
import os
import wandb
import glob





img_dir = '../dataset/train/images'
label_dir = '../dataset/train/labels'

print("🔍 Debug: Verifica contenuto cartella immagini")
for ext in ('*.jpg', '*.jpeg', '*.png'):
    files = glob.glob(os.path.join(img_dir, ext))
    print(f"Estensione {ext}: trovate {len(files)} immagini")

train_dataset = Custom_Dataloader(img_dir, label_dir, transforms_list=transforms_list)

print(f"Found {len(train_dataset)} images for training")
print(f"Example image path: {train_dataset.img_paths[0] if len(train_dataset) > 0 else 'None'}")

test_name ="yolov9s_multitransform_custom"
folder = "testing"

TrainerWithDataset = partial(CustomTrainer, custom_train_dataset=train_dataset)

wandb.init(project="YOLO_PROJECT", name=test_name, resume="allow")
 

model = YOLO("yolov9s.pt")
results = model.train(
    data="dataset.yaml",
    device=0,
    epochs=100,
    imgsz=640, 
    name=test_name,
    project = folder,
    trainer=TrainerWithDataset,
    plots = True,

    # 💡 Ottimizzazione
    optimizer="AdamW",           # Ottimizzatore robusto
    lr0=0.01,                    # Learning rate iniziale
    lrf=0.001,                   # LR minimo al termine (cosine decay)
    cos_lr=True,                 # Cosine learning rate decay
    momentum=0.937,
    weight_decay=0.0005,

    # 🔥 Warmup settings
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # 🧠 Generalizzazione e stabilità
    dropout=0.1,                 # Attivazione dropout
    patience=20,                 # Early stopping più tollerante

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