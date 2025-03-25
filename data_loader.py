import glob, cv2, torch, numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose(
    [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.BBoxSafeRandomCrop(width=640, height=640, p=0.5),
        A.Resize(640, 640),
        A.Normalize(),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0),
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

        augmented = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
        img_tensor = augmented["image"]
        boxes      = torch.tensor(augmented["bboxes"], dtype=torch.float32)
        labels     = torch.tensor(augmented["class_labels"], dtype=torch.long)

        # clamp e filter invalidi
        if boxes.numel():
            boxes[:,1:] = boxes[:,1:].clamp(0,1)
            valid = ((boxes[:,1:]>=0)&(boxes[:,1:]<=1)).all(dim=1)
            boxes, labels = boxes[valid], labels[valid]

        targets = torch.cat([labels.unsqueeze(1), boxes], dim=1) if boxes.numel() else torch.zeros((0,5))
        return img_tensor, targets

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    all_targets = []
    for i, t in enumerate(targets):
        if t.numel():
            batch_idx = torch.full((t.shape[0],1), i)
            all_targets.append(torch.cat([batch_idx, t], dim=1))
    all_targets = torch.cat(all_targets, 0) if all_targets else torch.zeros((0,6))
    return imgs, all_targets