import os
import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ultralytics import YOLO
from data_augmentation_custom import YOLODataset, collate_fn, train_transforms

# Percorsi
img_dir      = '/home/user/dataset/train/images'
label_dir    = '/home/user/dataset/train/labels'
dataset_yaml = 'dataset.yaml'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Carica class names
with open(dataset_yaml) as f:
    cfg = yaml.safe_load(f)
class_names = cfg.get("names", [])
print("Loaded class names")

# DataLoader
train_dataset = YOLODataset(img_dir, label_dir, transform=train_transforms)
print(f"Images found: {len(train_dataset.img_paths)}, Labels found: {len(train_dataset.label_paths)}")
dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Model
yolo = YOLO('yolov8n.yaml')
model = yolo.model.to(device)
model.names = class_names
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Target formatter
def format_targets(targets):
    batch_idx, cls, bboxes = [], [], []
    for i, t in enumerate(targets):
        if t.numel() == 0:
            continue
        labels = t[:, 0].long().unsqueeze(1)
        boxes  = t[:, 1:]
        batch_idx.append(torch.full((boxes.size(0),1), i, device=boxes.device))
        cls.append(labels.to(boxes.device))
        bboxes.append(boxes.to(boxes.device))
    if batch_idx:
        return {
            'batch_idx': torch.cat(batch_idx, dim=0),
            'cls':       torch.cat(cls, dim=0),
            'bboxes':    torch.cat(bboxes, dim=0)
        }
    return {'batch_idx': torch.zeros((0,1), device=device),
            'cls':       torch.zeros((0,1), device=device),
            'bboxes':    torch.zeros((0,4), device=device)}

# Training params
num_epochs = 40
precision_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        formatted = format_targets(targets)

        # Forward + loss in un solo passo
        result = model(images, formatted)

        loss       = result[0]
        loss_items = result[1] if len(result) > 1 else {}

        loss_scalar = loss.mean()       #trasformo il loss in uno scalare per la backward propagation

        optimizer.zero_grad()
        loss_scalar.backward()
        optimizer.step()

        epoch_loss += loss_scalar.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} — Loss medio: {avg_loss:.4f}")

    

    """-Validation--"""

    # Per il validation utilizzo  Ultralytics , ma prima devo creare un modello che utilizzi
    # i pesi che ho appena addestrato
    val_model = YOLO('yolov8n.yaml')
    val_model.model.load_state_dict(model.state_dict(), strict=False)
    results = val_model.val(data=dataset_yaml, imgsz=640, batch=16)

    if hasattr(results, "box") and hasattr(results.box, "pr"):
        precision = float(results.box.pr.mean())  # media su tutte le classi
    else:
        precision = 0.0
    
    print(f"Precision: {precision:.4f}")
    precision_history.append(precision)

# Save weights
save_path = os.path.join(os.getcwd(), "yolov8_manual_trained.pt")
torch.save(model.state_dict(), save_path)
print(f"Model saved at: {save_path}")

# Plot precision
plt.figure()
plt.plot(range(1, num_epochs+1), precision_history)
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Training Precision Over Epochs')
plt.grid(True)
plt.show()
