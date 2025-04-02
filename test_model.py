
from ultralytics import YOLO


# Inserire il nome del modello su cui effettuare il test finale
model_name = "custom_03"

path_to_bestweight = "/testing/"+model_name+"/weights/best.pt"

validation_name = model_name + "_validation"
folder = "test"

# Carica il modello con i pesi
model = YOLO(path_to_bestweight)

# Esegui inferenza sul video
metrics = model.val(
    data = "test_dataset.yaml",
    name=validation_name,
    project = folder,
    plot = True,  # Salva i grafici delle metriche
    save=True
)

print(f"▶️Test completato. Risultati salvati in {folder}/{validation_name}.")