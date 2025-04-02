# 📦 Importazioni standard e librerie per data augmentation e gestione immagini

import os
import glob
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === Definizione di più set di trasformazioni ===

# 🎨 Definizione di tre pipeline di trasformazione differenti (Albumentations)
# Ogni pipeline applica una combinazione diversa di augmentation per aumentare la varietà dei dati
# Utilizzano Resize a 640x640 e il formato YOLO per i bounding box

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
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
    A.Perspective(p=0.3),
    A.HorizontalFlip(p=0.3),
    A.Resize(640, 640),
    ToTensorV2(),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0))

transform_set3 = A.Compose([
    A.SomeOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1),
        A.MotionBlur(blur_limit=5, p=1),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1),
        A.OpticalDistortion(
            distort_limit=(-0.05,0.05),
            interpolation=1,
            mask_interpolation=0,
            mode = 'camera',
            p=0.3
        ),
        A.Affine(
            translate_percent={"x": 0.05, "y": 0.05},  # traslazione max ±5%
            scale=(0.9, 1.1),                          # scala 0.9x–1.1x
            rotate=(-10, 10),                          # rotazione ±10°
            shear={"x": (-5, 5), "y": (-5, 5)},        # opzionale
            p=1.0
        ),
        A.Perspective(p=1),
        A.RandomCrop(height=512, width=512, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
    ], n=2, replace=False, p=0.8),  # Applica 2 trasformazioni casuali su 5
    A.Resize(640, 640),
    ToTensorV2(),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0))

# 🔁 Elenco delle trasformazioni da applicare
transforms_list = [
    transform_set1,
    transform_set2,
    transform_set3,
]


# 🧱 Definizione del Custom Dataloader multi-trasformazione
# Questo DataLoader applica ogni trasformazione presente in `transforms_list` a ogni immagine.
# L'output di __getitem__ è una lista di tuple: una per ogni versione augmentata.

class Custom_Dataloader(Dataset):
    def __init__(self, img_dir, label_dir, transforms_list,num_transforms):

         # 🔍 Raccoglie tutti i file immagine con estensioni valide
        self.img_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.img_paths.extend(sorted(glob.glob(os.path.join(img_dir, ext))))

         # 📄 Raccoglie i file etichetta YOLO-style (.txt)
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        self.transforms_list = transforms_list

        # 📊 Statistiche per il debug/report
        self.skipped_augmentation = 0
        self.empty_targets = 0
        self.total_samples = 0
        self.num_transforms = num_transforms

    # Restituisce il numero totale di immagini nel dataset
    def __len__(self):
        return len(self.img_paths)
    
    # 🔍 CONTROLLO IMMAGINE NERA POST-TRANSFORMAZIONE
    def is_black_image(image: torch.Tensor, std_thresh=2.0, max_thresh=20) -> bool:
        """
        Considera l'immagine nera se la varianza è bassa e il valore massimo è sotto soglia.
        Questo approccio evita falsi positivi su immagini FLIR scure ma con contenuto utile.
        """
        img_np = image.detach().cpu().numpy()

        if img_np.shape[0] == 3:
            gray = 0.2989 * img_np[0] + 0.5870 * img_np[1] + 0.1140 * img_np[2]
        else:
            gray = img_np[0]  # Se già grayscale

        std_val = gray.std()
        max_val = gray.max()

        # Debug (solo per test temporaneo)
        print(f"[DEBUG] std={std_val:.2f} | max={max_val:.2f}")
        return std_val < std_thresh and max_val < max_thresh

    def __getitem__(self, idx):

        # 🔄 Metodo principale per caricare e trasformare un'immagine
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        # Carica immagine e converte in RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

         # 📌 Parsing delle etichette YOLO (classe, x, y, w, h)
        bboxes, class_labels = [], []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(cls))

        results = []

         # 🔁 Per ogni trasformazione definita, applica l'augmentation e valida i bounding box
        for i in range(self.num_transforms):
            try:
                transform = random.choice(self.transforms_list)  # Scegli una trasformazione casuale
                # 🔄 Applica la trasformazione
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)

                # ✅ Validazione post-transform: solo box con coordinate valide (in [0,1])
                valid_bboxes, valid_labels = [], []
                for box, label in zip(transformed["bboxes"], transformed["class_labels"]):
                    if len(box) == 4 and all(0.0 <= v <= 1.0 for v in box):
                        valid_bboxes.append(box)
                        valid_labels.append(label)
            except Exception:
                 # ❌ Se Albumentations lancia errore, logga e restituisce immagine vuota
                self.skipped_augmentation += 1
                img_tensor = torch.zeros((3, 640, 640))
                targets = torch.zeros((0, 5))
                results.append((img_tensor, targets, img_path))
                continue
            
            # 🔢 Conversione in tensor
            img_tensor = transformed["image"]
            boxes = torch.tensor(valid_bboxes, dtype=torch.float32)
            labels = torch.tensor(valid_labels, dtype=torch.long)

             # 🎯 Filtro di box troppo piccoli o non validi
            if boxes.numel():
                valid = (boxes[:, 2] > 0.01) & (boxes[:, 3] > 0.01)
                boxes = boxes[valid]
                labels = labels[valid]

             # 📈 Conta i target vuoti
            if boxes.numel() == 0:
                self.empty_targets += 1

             # 📦 Organizza target come tensor [cls, x, y, w, h]
            targets = torch.cat([labels.unsqueeze(1), boxes], dim=1) if boxes.numel() else torch.zeros((0, 5))
            results.append((img_tensor, targets, img_path))

        self.total_samples += 1
        return results  # lista di tuple (img_tensor, targets, path) per ogni trasformazione

    def report_stats(self):
        print("\n=== Report Dataset ===")
        print(f"Totale immagini processate: {self.total_samples}")
        print(f"Immagini con 0 target validi: {self.empty_targets}")
        print(f"Immagini saltate per errore Albumentations: {self.skipped_augmentation}")
        valid_samples = self.total_samples - self.empty_targets - self.skipped_augmentation
        print(f"Totale immagini realmente utilizzate per il training: {valid_samples}")
        print("======================\n")

    def reset(self):
        """📌 Metodo richiesto da Ultralytics per compatibilità (resetta le metriche interne)."""
        print("📌 Custom_Dataloader.reset() called")
        self.skipped_augmentation = 0
        self.empty_targets = 0
        self.total_samples = 0
    
    


# ===collate_fn ===

# 🧩 Funzione collate personalizzata
# Viene usata per appiattire la lista di immagini trasformate multiple in un unico batch PyTorch.
# La funzione collate_fn è fondamentale per combinare i dati in batch durante il training.

def collate_fn(batch):

    # 🔄 Appiattisce tutti i risultati delle trasformazioni in un'unica lista
    flat_batch = [item for sublist in batch for item in sublist]  # flattens list of lists
    imgs, targets, paths = zip(*flat_batch)

    imgs = torch.stack(imgs)

     # 📦 Prepara tensori batch_idx, cls e bbox per l'interfaccia YOLO
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
        # ⛔ Nessun target valido
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
