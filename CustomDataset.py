import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, mode='train', root_dir=None, list_file=None, transform=None, label_folder_name='labels'):
        """
        Args:
            mode: 'train' oppure 'val' (o 'test') per indicare quale set utilizzare.
            root_dir: cartella principale del dataset
            list_file: per mode 'val' (o 'test') il file di testo che contiene i percorsi delle immagini.
            transform: eventuali trasformazioni (in futuro potremo aggiungere data augmentation).
            label_folder_name: nome della cartella dove sono salvati i file di label. 
        """
        
        self.mode = mode
        self.transform = transform
        self.label_folder_name = label_folder_name

        # a seconda della mode cambia dove vengono prelevate le immagini (già normalizzate con convert_annotations)

        if mode == 'train':
            # Per il training le immagini sono in root_dir/thermal_8_bit e le label in root_dir/labels
            self.image_dir = os.path.join(root_dir, 'thermal_8_bit')
            self.label_dir = os.path.join(root_dir, label_folder_name)
            # Creiamo una lista di percorsi delle immagini
            self.image_paths = sorted([os.path.join(self.image_dir, f) 
                                        for f in os.listdir(self.image_dir) 
                                        if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
        else:
            # Per validation/test i percorsi sono letti dal file di listing
            with open(list_file, 'r') as f:
                self.image_paths = [line.strip() for line in f.readlines()]
            # Per le label, assumiamo che siano in root_dir/labels
            self.label_dir = os.path.join(root_dir, label_folder_name)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Carica l'immagine con PIL
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Costruisci il percorso della label: prendi il nome base dell'immagine e sostituisci l'estensione con .txt
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base_name + '.txt')
        
        # Leggi il file delle label (formato YOLO: classe, x_center, y_center, width, height)
        bboxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_center, y_center, width, height = parts
                    classes.append(int(cls))
                    bboxes.append([float(x_center), float(y_center), float(width), float(height)])
        else:
            # Se non esiste il file label, restituisce liste vuote
            bboxes = []
            classes = []
        
        # Prepara il sample con immagine, bounding box, classi e il percorso dell'immagine (utile per debug)
        sample = {'image': image, 'bboxes': bboxes, 'classes': classes, 'img_path': img_path}
        
        # Applica eventuali trasformazioni (ad esempio, data augmentation in seguito)
        if self.transform:
            sample = self.transform(sample)
        else:
            # Se non ci sono trasformazioni, convertiamo l'immagine in tensore (valori normalizzati in [0,1])
            image_np = np.array(image)
            # Convertiamo da HxWxC a CxHxW
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            sample['image'] = image_tensor
        
        return sample

# --- Esempi di utilizzo del DataLoader ---

if __name__ == "__main__":
    # DataLoader per il training
    train_root = 'C:/Users/Marco/Desktop/Nuova cartella/dataset/train'
    train_dataset = CustomDataset(mode='train', root_dir=train_root, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print("Training batch:")
    for batch in train_loader:
        print("Shape immagini:", batch['image'].shape)
        print("Label (bounding boxes):", batch['bboxes'])
        print("Classi:", batch['classes'])
        print("Percorsi immagini:", batch['img_path'])
        break  # Visualizza solo il primo batch

    # DataLoader per validation/test usando il file di listing
    # Scegli ad esempio il file "val_test.txt" per il test
    val_root = 'C:/Users/Marco/Desktop/Nuova cartella/dataset/val'
    list_file = 'C:/Users/Marco/Desktop/Nuova cartella/dataset/val_test.txt'
    val_dataset = CustomDataset(mode='val', root_dir=val_root, list_file=list_file, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print("\nValidation/Test batch:")
    for batch in val_loader:
        print("Shape immagini:", batch['image'].shape)
        print("Label (bounding boxes):", batch['bboxes'])
        print("Classi:", batch['classes'])
        print("Percorsi immagini:", batch['img_path'])
        break  # Visualizza solo il primo batch
