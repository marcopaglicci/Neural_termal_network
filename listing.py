import os
import glob
import random

"""
Questo script ha il compito di dividere il dataset della cartella val in 2 elementi per 
la validazione e il testing, per fare questo e rendere il dataloader modulare suddivide
i due elementi casualmente e salva su un file di txt i path per ogni immagine di test , e su un'altro
i path di ogni immagine di validation. invece che creare 2 cartelli nuove mantiene la "struttura"
del dataset così com'è presente.
"""

# Imposta i percorsi corretti per le immagini in val e per le destinazioni dei file di listing
val_images_dir = 'C:/Users/Marco/Desktop/Nuova cartella/dataset/val/thermal_8_bit'
output_val_txt = 'C:/Users/Marco/Desktop/Nuova cartella/dataset/val_validation.txt'
output_test_txt = 'C:/Users/Marco/Desktop/Nuova cartella/dataset/val_test.txt'

# Recupera la lista di immagini
images = glob.glob(os.path.join(val_images_dir, '*.jpeg'))
images.sort()  # Ordina per avere una base consistente
random.seed(42)  # Per riproducibilità
random.shuffle(images)

# Dividi la lista; ad esempio, il 50% per validazione e il 50% per test
split_ratio = 0.5
split_index = int(len(images) * split_ratio)
validation_images = images[:split_index]
test_images = images[split_index:]

# Salva la lista di validation
with open(output_val_txt, 'w') as f:
    for img_path in validation_images:
        f.write(f"{img_path}\n")

# Salva la lista di test
with open(output_test_txt, 'w') as f:
    for img_path in test_images:
        f.write(f"{img_path}\n")

print("File lists per validation e test creati correttamente.")
