
import glob
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms  = A.Compose(
    [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.BBoxSafeRandomCrop(p=0.5),
        A.Resize(640, 640),            # fa il resize delle immagini nel formato richesto 
        A.Normalize(),                 # esegue la normalizzazione per velocizzare il training 
        ToTensorV2(),                  # Tensor PyTorch pronto per essere passato al modello 
    ],  
     bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.0,           # conserva anche box molto piccoli
    )
)

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform):
        self.img_paths   = sorted(glob.glob(f"{img_dir}/*.jpeg"))
        self.label_paths = sorted(glob.glob(f"{label_dir}/*.txt"))
        self.transform   = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])[..., ::-1]
        bboxes, class_labels = [], []
        for line in open(self.label_paths[idx]):
            cls, x, y, w, h = map(float, line.split())
            bboxes.append([x, y, w, h])
            class_labels.append(int(cls))

        try:
            augmented = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
            img_tensor = augmented["image"]
            boxes      = torch.tensor(augmented["bboxes"], dtype=torch.float32)
            labels     = torch.tensor(augmented["class_labels"], dtype=torch.long)
        except ValueError:
            # Fallback manual
            img_resized = cv2.resize(img, (640, 640))
            img_tensor  = ToTensorV2()(image=img_resized.astype(np.float32) / 255.0)["image"]
            boxes       = torch.tensor(bboxes, dtype=torch.float32)
            labels      = torch.tensor(class_labels, dtype=torch.long)

        # Clamp box coords e filtra invalidi
        if boxes.numel() > 0:
            boxes[:, 1:] = boxes[:, 1:].clamp(0.0, 1.0)
            valid = ((boxes[:,1:] >= 0) & (boxes[:,1:] <= 1)).all(dim=1)
            boxes  = boxes[valid]
            labels = labels[valid]

        targets = torch.cat([labels.unsqueeze(1), boxes], dim=1) if boxes.numel() else torch.zeros((0,5))
        return img_tensor, targets


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return torch.stack(imgs), targets
