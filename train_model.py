# 📦 Import delle librerie necessarie
# Si importano il modello YOLO da Ultralytics e os per la gestione dei path

from ultralytics import YOLO, settings
import os


# 🧠 Inizializzazione del modello
# Carica il modello pre-addestrato YOLOv9s (.pt) con pesi già pronti per il fine-tuning
model = YOLO("yolov9s.pt")

# ℹ️ Stampa un riepilogo della rete (numero layer, parametri, input shape, ecc.)
model.info()

# 🏷️ Configurazione dell’esperimento
# Definisce il nome con cui verrà salvata la sessione di training e la cartella di output
test_name = "test_8-custom_augmentation_9"
folder = "testing"


# 🚀 Avvio del training del modello
# Specifica i parametri principali del ciclo di addestramento, inclusa:
# - auto_augment: modalità automatica di data augmentation integrata in Ultralytics
# - optimizer: usa l’ottimizzatore SGD con learning rate iniziale definito
# - resume=True: consente di riprendere da checkpoint se esistenti
results = model.train(
    data="dataset.yaml",       
    epochs=100,                
    imgsz=640,                 
    device=0,
    optimizer='SGD',
    auto_augment='autoaugment',
    resume = True,
    name = test_name,
    project = folder,
    lr0=0.01,        
    lrf=0.0,         
    plots = True
)

# ✅ Messaggi di conferma a fine training
# Conferma la fine dell'addestramento e la directory in cui sono salvati i risultati

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


metrics = model.val()
