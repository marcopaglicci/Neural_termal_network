from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch



# Carica il modello (qui usiamo il modello YOLOv8n pre-addestrato)
model = YOLO("yolov8n.pt")

# Avvia il training specificando i file di configurazione
# Puoi aggiungere ulteriori flag come --img 640 se vuoi specificare la risoluzione
results = model.train(
    data="dataset.yaml",       # File YAML del dataset
    epochs=70,                # Numero di epoche
    imgsz=640,                  # Dimensione immagine (opzionale, di default YOLOv8 gestisce il letterboxing)
    device=0,
    lr0=0.01,
    lrf = 0.1,
    momentum=0.937,
    optimizer='AdamW',
    auto_augment='randaugment'
)

# YOLOv8 salva i risultati del training in una directory (di solito runs/train/exp*)
# Se l'oggetto results possiede l'attributo "path", lo usiamo per individuare la directory di log:

train_dir = results.path if hasattr(results, "path") else "runs/train/exp"

print(f"Training completato. I risultati sono salvati in: {train_dir}")

# Prova a usare la funzione plot_results integrata in YOLOv8 (se disponibile)

print("Funzione plot_results non disponibile, eseguo un plotting manuale.")
# Carica il file CSV dei risultati (solitamente results.csv nella directory di training)
csv_path = os.path.join(train_dir, "results.csv")
if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()
else:
    print(f"File CSV dei risultati non trovato in: {csv_path}")
