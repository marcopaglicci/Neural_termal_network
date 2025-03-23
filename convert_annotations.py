import os
import json


"""convert_bbox prende come input le dimensioni dell'immagine (size, ad es. (width, height)) 
e una bounding box nel formato COCO [x, y, w, h] (valori assoluti in pixel) e restituisce le 
coordinate della box normalizzate (cioè, espresse in percentuale rispetto alla dimensione dell'immagine).

Questo perchè YOLO utilizza coordinate delle bounding box normalizzate, il dataset ha label in formato COCO e
dobbiamo renderlo compatibile con il formato YOLO"""

def convert_bbox(size, box):
    """
    Converte un bounding box da formato COCO (x, y, w, h) a YOLO (x_center, y_center, w, h) normalizzati.
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x, y, w, h = box
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    return (x_center, y_center, w, h)


"""prelevo il path del dataset di immagini dalla cartella train"""

conversion_directory = "val"
json_directory = 'C:/Users/Marco/Desktop/Nuova cartella/dataset/val/'
train_dir = 'C:/Users/Marco/Desktop/dataset/'+ conversion_directory
image_folder  = os.path.join(train_dir, 'images')
json_path = os.path.join(json_directory, 'thermal_annotations.json')
output_folder = os.path.join(train_dir, 'labels')
os.makedirs(output_folder, exist_ok=True)


"""----carico il file JSON contenente le annotazioni----"""
# Carica il file JSON
with open(json_path, 'r') as f:
    data = json.load(f)


"""
Mappo ogni immagine del dataset alle sue informazioni relative
e raggruppo le annotazioni per ogni immagine in modo da poter accedere
facilmente a tutte le annotazioni presenti su una img
"""

#raggruppo le immagini in base all'id
images = {img['id']: img for img in data['images']}

# Raggruppa le annotazioni per immagine attraverso un dizionario
annotations_by_image = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    annotations_by_image.setdefault(img_id, []).append(ann)



"""
Per ciascuna immagine elencata nel JSON, il codice costruisce il percorso completo usando il campo "file_name".
Se limmagine esiste nella cartella del dataset, si procode alla conversione
    
"""

# Itera sulle immagini e crea il file di annotazioni YOLO
for img_id, img_info in images.items():
    img_filename = img_info['file_name']

    # Verifica che l'immagine esista nella cartella di immagini termiche annotate
    img_path = os.path.join(train_dir, img_info['file_name'])
    if not os.path.isfile(img_path):
        print(f"File non trovato: {img_path}")
        continue  # Salta se l'immagine non è presente in questa cartella

    #estraggo le informazioni
    width = img_info['width']
    height = img_info['height']
    txt_filename = os.path.splitext(os.path.basename(img_info['file_name']))[0] + '.txt'
    txt_path = os.path.join(output_folder, txt_filename)
    
    with open(txt_path, 'w') as out_file:
        # Se ci sono annotazioni per questa immagine, processale
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                class_id = ann['category_id']   
                bbox = ann['bbox']  # [x, y, width, height]
                x_center, y_center, w_norm, h_norm = convert_bbox((width, height), bbox)
                out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

"""Salvo le annotazioni in un file associato di testo dentro la cartella Labels, cosi da poterle utilizzare"""

print("Conversione completata. I file di annotazioni YOLO sono stati salvati nella cartella:", output_folder)
