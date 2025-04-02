
from ultralytics import YOLO
import os

# Inserire il nome del video su cui effettuare predict
video_name = "video01"
model_name = "custom_03"

path_to_bestweight = "/testing/"+model_name+"/weights/best.pt"



# Carica il modello con i pesi
model = YOLO(path_to_bestweight)

# Percorso al video da analizzare
path_to_video = f"../video/{video_name}.mp4"

# Esegui inferenza sul video
results = model.predict(
    source=path_to_video,
    conf=0.25,
    device=0,
    save=True,    # salva video con predizioni in runs/
    show=True     # opzionale: mostra in tempo reale possibile solo se la macchina di prova supporta GUI
)

print(f"▶️ Video inferenza completata.")