import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Transformazioni con Albumentations
custom_transforms = A.Compose(
    [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
        A.Perspective(p=0.3),
        A.Blur(p=0.2),
        A.GaussNoise(p=0.2),
        A.Resize(640, 640),  # Questo serve per compatibilità YOLO
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0),
)

class Custom_Dataloader(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.img_paths.extend(sorted(glob.glob(os.path.join(img_dir, ext))))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        self.transform = transform
        self.skipped_augmentation = 0
        self.empty_targets = 0
        self.total_samples = 0

        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = [], []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(cls))

        if self.transform:
            try:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
                valid_bboxes, valid_labels = [], []
                for box, label in zip(transformed["bboxes"], transformed["class_labels"]):
                    if len(box) == 4 and all(0.0 <= v <= 1.0 for v in box):
                        valid_bboxes.append(box)
                        valid_labels.append(label)
            except Exception as e:
                self.skipped_augmentation += 1
                return torch.zeros((3, 640, 640)), torch.zeros((0, 5)), img_path
            
            img_tensor = transformed["image"]
            boxes = torch.tensor(valid_bboxes, dtype=torch.float32)
            labels = torch.tensor(valid_labels, dtype=torch.long)
            
            
        else:
            img_tensor = ToTensorV2()(image=img)["image"]
            boxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(class_labels, dtype=torch.long)

        # Filtra bbox troppo piccoli
        if boxes.numel():
            valid = (boxes[:, 2] > 0.01) & (boxes[:, 3] > 0.01)
            boxes = boxes[valid]
            labels = labels[valid]

        if boxes.numel() == 0:
            self.empty_targets += 1

        # Costruzione targets
        self.total_samples += 1
        targets = torch.cat([labels.unsqueeze(1), boxes], dim=1) if boxes.numel() else torch.zeros((0, 5))
        return img_tensor, targets,self.img_paths[idx]


    def report_stats(self):
        print("\n=== Report Dataset ===")
        print(f"Totale immagini processate: {self.total_samples}")
        print(f"Immagini con 0 target validi: {self.empty_targets}")
        print(f"Immagini saltate per errore Albumentations: {self.skipped_augmentation}")
        valid_samples = self.total_samples - self.empty_targets - self.skipped_augmentation
        print(f"Totale immagini realmente utilizzate per il training: {valid_samples}")
        print("======================\n")


def collate_fn(batch):
    imgs, targets,paths  = zip(*batch)
    imgs = torch.stack(imgs)

    if any(t.numel() > 0 for t in targets):
        batch_idx = []
        cls, bboxes = [], []
        for i, t in enumerate(targets):
            batch_idx.append(torch.full((t.shape[0],), i))
            cls.append(t[:, 0])
            bboxes.append(t[:, 1:5])
        batch_idx = torch.cat(batch_idx)
        cls = torch.cat(cls)
        bboxes = torch.cat(bboxes)
    else:
        batch_idx = torch.zeros((0,), dtype=torch.long)
        cls = torch.zeros((0,), dtype=torch.long)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)


    return {
        "img": imgs,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "im_file": list(paths)
    }
