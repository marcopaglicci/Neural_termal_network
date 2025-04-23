# Fine-Tuning YOLO for Thermal Object Detection 🔥📦

**Autore:** Marco Paglicci  
**Università:** Università degli Studi di Firenze - Ingegneria Informatica  
**Anno Accademico:** 2025  
**Relatore:** Prof. Marco Bertini

---

## 📌 Descrizione del progetto

Questo progetto implementa un sistema di rilevazione oggetti su **immagini termiche** basato su modelli **YOLOv9s** fine-tuned, sfruttando un framework personalizzato di **data augmentation** con Albumentations e un trainer customizzato basato sulla libreria Ultralytics.

L'obiettivo è stato adattare un modello pre-addestrato YOLO per il dominio termico, migliorandone le prestazioni grazie a:

- Architettura YOLOv9s
- Trainer personalizzato (`CustomTrainer`)
- Data augmentation avanzata (multi-pipeline)
- Validazione su sottoinsiemi non visti
- Applicazione su video termici reali

---

## 🧠 Tecnologie utilizzate

- [Ultralytics YOLOv8/YOLOv9](https://docs.ultralytics.com/)
- [PyTorch 2.6](https://pytorch.org/)
- [Albumentations](https://albumentations.ai/)
- OpenCV, NumPy, Matplotlib
- Dataset: [Teledyne FLIR Thermal Dataset](https://www.flir.com/oem/adas/adas-dataset-form/)

---


## ⚙️ Requisiti

- Python >= 3.9
- CUDA compatibile (GPU Nvidia)
- torch >= 2.0
- ultralytics >= 8.3.95
- albumentations >= 1.3.1

## 🚀 Come eseguire il training
Prepara il dataset:
-Organizza le immagini in formato YOLO (img + label .txt)
-Aggiorna dataset.yaml con i path locali

Avvia il training (in bash):

python custom_trainer_model.py

