import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === Definizione di più set di trasformazioni ===
transform_set1 = A.Compose([
    A.RandomBrightnessContrast(p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
        A.GaussianBlur(blur_limit=3),
    ], p=0.3),
    A.Resize(640, 640),
    ToTensorV2()
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0))

transform_set2 = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.7),
    A.Perspective(p=0.7),
    A.Resize(640, 640),
    ToTensorV2(),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0))

transform_set3 = A.Compose([
    A.GaussNoise(var_limit=(5, 15), p=0.3),
    A.HorizontalFlip(p=0.3),
    A.Resize(640, 640),
    ToTensorV2(),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0))

transforms_list = [transform_set1, transform_set2, transform_set3]


# === Custom Dataloader aggiornato ===
class Custom_Dataloader(Dataset):
    def __init__(self, img_dir, label_dir, transforms_list):
        self.img_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.img_paths.extend(sorted(glob.glob(os.path.join(img_dir, ext))))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        self.transforms_list = transforms_list
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

        results = []
        for transform in self.transforms_list:
            try:
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                valid_bboxes, valid_labels = [], []
                for box, label in zip(transformed["bboxes"], transformed["class_labels"]):
                    if len(box) == 4 and all(0.0 <= v <= 1.0 for v in box):
                        valid_bboxes.append(box)
                        valid_labels.append(label)
            except Exception:
                self.skipped_augmentation += 1
                img_tensor = torch.zeros((3, 640, 640))
                targets = torch.zeros((0, 5))
                results.append((img_tensor, targets, img_path))
                continue

            img_tensor = transformed["image"]
            boxes = torch.tensor(valid_bboxes, dtype=torch.float32)
            labels = torch.tensor(valid_labels, dtype=torch.long)

            if boxes.numel():
                valid = (boxes[:, 2] > 0.01) & (boxes[:, 3] > 0.01)
                boxes = boxes[valid]
                labels = labels[valid]

            if boxes.numel() == 0:
                self.empty_targets += 1

            targets = torch.cat([labels.unsqueeze(1), boxes], dim=1) if boxes.numel() else torch.zeros((0, 5))
            results.append((img_tensor, targets, img_path))

        self.total_samples += 1
        return results  # lista di tuple per ogni trasformazione

    def report_stats(self):
        print("\n=== Report Dataset ===")
        print(f"Totale immagini processate: {self.total_samples}")
        print(f"Immagini con 0 target validi: {self.empty_targets}")
        print(f"Immagini saltate per errore Albumentations: {self.skipped_augmentation}")
        valid_samples = self.total_samples - self.empty_targets - self.skipped_augmentation
        print(f"Totale immagini realmente utilizzate per il training: {valid_samples}")
        print("======================\n")

    def reset(self):
        """Metodo fittizio per compatibilità con Ultralytics."""
        print("📌 Custom_Dataloader.reset() called")
        self.skipped_augmentation = 0
        self.empty_targets = 0
        self.total_samples = 0
    
    


# === Nuova collate_fn ===
def collate_fn(batch):
    # Appiattisce tutte le trasformazioni generate per ogni immagine in un'unica lista
    flat_batch = [item for sublist in batch for item in sublist]  # flattens list of lists
    imgs, targets, paths = zip(*flat_batch)

    imgs = torch.stack(imgs)
    if any(t.numel() > 0 for t in targets):
        batch_idx, cls, bboxes = [], [], []
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
