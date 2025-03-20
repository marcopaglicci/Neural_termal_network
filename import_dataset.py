import os
import json
from PIL import Image
import matplotlib.pyplot as plt

# Specifica il percorso della cartella 'train' del tuo dataset FLIR
train_dir = 'C:/Users/Marco/Desktop/Nuova cartella/dataset/train'

# Percorso della cartella con le immagini termiche annotate (modifica se il nome della cartella è diverso)
annotated_folder = os.path.join(train_dir, 'Annotated_thermal_8_bit')

# Verifica l'esistenza della cartella
if not os.path.isdir(annotated_folder):
    raise FileNotFoundError(f"La cartella {annotated_folder} non esiste. Verifica il percorso e il nome della cartella.")

# Ottieni la lista di file immagine (filtra per estensioni comuni)
image_extensions = ('.png', '.jpg', '.jpeg')
image_files = [f for f in os.listdir(annotated_folder) if f.lower().endswith(image_extensions)]

print(f"Trovate {len(image_files)} immagini termiche annotate.")

# Carica un'immagine di esempio
if image_files:
    image_path = os.path.join(annotated_folder, image_files[0])
    img = Image.open(image_path)
    
    # Visualizza l'immagine con matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.title("Esempio di immagine termica")
    plt.axis('off')
    plt.show()
else:
    print("Nessuna immagine trovata nella cartella.")

# Carica il file JSON delle annotazioni
json_path = os.path.join(train_dir, 'thermal_annotations.json')
if os.path.isfile(json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    print("Chiavi presenti nel file di annotazioni:", list(annotations.keys()))
else:
    print(f"Il file {json_path} non esiste. Verifica il percorso.")
