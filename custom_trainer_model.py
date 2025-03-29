# 📦 Importazione dei moduli principali
# Importa le classi per il training, il modello YOLO, i trasformatori personalizzati,
# e le librerie per logging, gestione file e dataset

from functools import partial
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from data_loader import YOLODataset , train_transforms
#from  Custom_Dataloader import Custom_Dataloader, custom_transforms
from custom_trainer import CustomTrainer
from MultiTransform_Dataloader import Custom_Dataloader, transforms_list
import os
import wandb
import glob


# 📂 Setup del dataset
# Definisce i percorsi delle immagini e delle etichette del dataset di training.
# Inoltre, effettua una scansione dei file presenti per verificarne il contenuto

img_dir = '../dataset/train/images'
label_dir = '../dataset/train/labels'

print("🔍 Debug: Verifica contenuto cartella immagini")
for ext in ('*.jpg', '*.jpeg', '*.png'):
    files = glob.glob(os.path.join(img_dir, ext))
    print(f"Estensione {ext}: trovate {len(files)} immagini")

# 🔁 Inizializzazione del dataloader personalizzato
# Viene istanziato un Custom_Dataloader, capace di applicare trasformazioni multiple per ogni immagine.
# Questo approccio arricchisce il dataset con versioni augmentate, potenziando la generalizzazione del modello.

train_dataset = Custom_Dataloader(img_dir, label_dir, transforms_list=transforms_list)

print(f"Found {len(train_dataset)} images for training")
print(f"Example image path: {train_dataset.img_paths[0] if len(train_dataset) > 0 else 'None'}")

# 🛠️ Configurazione dell’esperimento
# Specifica il nome dell’esperimento, la directory di output e inizializza WandB per il tracking

test_name ="yolov9s_multitransform_custom_03"
folder = "testing"

TrainerWithDataset = partial(CustomTrainer, custom_train_dataset=train_dataset)

wandb.init(project="YOLO_PROJECT", name=test_name, resume="allow")
 
# 🚀 Avvio del training del modello YOLOv9s
# Il modello viene caricato e addestrato utilizzando un trainer personalizzato che sfrutta
# il dataloader custom. Sono specificate anche varie opzioni per l’ottimizzazione.

model = YOLO("yolov9s.pt")
results = model.train(
    data="dataset.yaml",         # Specifica il file YAML per il dataset
    device=0,                    # Training su GPU
    epochs=100,
    imgsz=640, 
    name=test_name,
    project = folder,
    trainer=TrainerWithDataset,  # Trainer custom con dataloader personalizzato
    plots = True,
    

    # 💡 Ottimizzazione
    optimizer="AdamW",           # Ottimizzatore robusto
    lr0=0.005,                   # Learning rate iniziale
    lrf=0.001,                   # LR minimo al termine (cosine decay)
    cos_lr=True,                 # Cosine learning rate decay
    momentum=0.937,
    weight_decay=0.002,

    # 🔥 Warmup settings
    warmup_epochs=10.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # 🧠 Generalizzazione e stabilità
    #dropout=0.2,                 # Attivazione dropout
                    
)

# 📊 Analisi delle statistiche di training
# Dopo il training viene stampato un sommario delle statistiche del dataset (implementazione lato Custom_Dataloader)

train_dataset.report_stats()

print(test_name + " - Training complete ")
print("All  result saved in " + folder )


# 💾 Salvataggio dei pesi migliori
# Costruisce il path dei pesi migliori ottenuti durante il training

best_weights = os.path.join(folder, "weights", "best.pt")
print(f"Best weights  for "  +  test_name + " saved at: {best_weights}")


# ✅ Validazione del modello addestrato
# Viene avviata una fase di validazione e salvati i risultati in una sottodirectory dedicata

validation_name = test_name + "_validation"
metrics = model.val(
    name=validation_name,
    project = folder,
)

print(validation_name + " - Validation complete ")
print("All  result saved in " + folder + "/validation_name ")