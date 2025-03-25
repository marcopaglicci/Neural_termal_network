from functools import partial
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from data_loader import YOLODataset , train_transforms
from custom_trainer import CustomTrainer
import os


img_dir = '/home/user/dataset/train/images'
label_dir = '/home/user/dataset/train/labels'
train_dataset = YOLODataset(img_dir, label_dir, train_transforms)

test_name = "yolov8s_randaugment_deafult"
folder = "testing"

TrainerWithDataset = partial(CustomTrainer, custom_train_dataset=train_dataset)
 

model = YOLO("yolov8s.pt")
results = model.train(
    data="dataset.yaml",
    device=0,
    epochs=100,
    imgsz=640,      
    name=test_name,
    project = folder,
    auto_augment = 'randaugment',
    plots = True
)

print(test_name + " - Training complete ")
print("All  result saved in " + folder )

best_weights = os.path.join(results.path, "weights", "best.pt")
print(f"Best weights saved at: {best_weights}")

validation_name = test_name + "_validation"
metrics = model.val(
    name=validation_name,
    project = folder,
    plot = True
)

print(validation_name + " - Validation complete ")
print("All  result saved in " + folder + "/validation_name ")