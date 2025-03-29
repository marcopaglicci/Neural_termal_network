# 📦 Importazione delle librerie necessarie
# Importa il trainer di base di Ultralytics, DataLoader di PyTorch,
# e il collate_fn personalizzato per la gestione batch dei target YOLO-style

from functools import partial
from ultralytics.models.yolo.detect.train import DetectionTrainer
from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from torch.utils.data import DataLoader
from MultiTransform_Dataloader import  collate_fn  
import logging

# 📝 Logging configurato per tracciare le fasi critiche di inizializzazione e caricamento

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
   
)
logger = logging.getLogger(__name__)

# ⚙️ Definizione del Trainer personalizzato
# Questa classe estende DetectionTrainer per integrare un dataset personalizzato
# e sostituire il comportamento standard di caricamento dei dati con logica custom.

class CustomTrainer(DetectionTrainer):


    def __init__(self, *args, custom_train_dataset=None, **kwargs):
        logger.info("CustomTrainer initialization started")
        super().__init__(*args, **kwargs)
        self.custom_train_dataset = custom_train_dataset

         # Disattiva la stampa delle label durante il training (evita plot automatici)
        self.plot_training_labels = lambda *a, **kw: None

        logger.info(f"CustomTrainer initialized with dataset: {custom_train_dataset is not None}")


    # 🔁 Override del metodo get_dataloader
    # Permette di sostituire il dataset standard con quello custom quando si è in modalità training.

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        logger.info(f"get_dataloader called with mode={mode}")
        if mode == "train" and self.custom_train_dataset is not None:
            logger.info("Using custom dataset in get_dataloader")
            return self.build_dataloader(self.custom_train_dataset, mode=mode, batch=batch_size)
        
        logger.info("Falling back to parent get_dataloader")
        return super().get_dataloader(dataset_path, batch_size, rank, mode)
    

    # 🧱 Metodo per costruire il dataloader
    # Crea un DataLoader PyTorch con il dataset personalizzato, abilitando opzioni come shuffle, pin_memory e workers.
    # In caso di errore, stampa un messaggio dettagliato e rilancia l’eccezione.

    def build_dataloader(self, dataset, mode="train", batch=None):
        logger.info(f"build_dataloader called with mode={mode}, batch={batch}")
        logger.info(f"Dataset type: {type(dataset)}")

        if mode == "train" and self.custom_train_dataset is not None:
            logger.info(f"Using custom dataset for training with {len(self.custom_train_dataset)} samples")
            try:
                dataloader = DataLoader(
                    self.custom_train_dataset,
                    batch_size=self.args.batch if batch is None else batch,
                    shuffle=True,
                    num_workers=self.args.workers,
                    collate_fn=collate_fn,
                    pin_memory=True
                )
                logger.info(f"Successfully created dataloader with {len(dataloader)} batches")
                dataloader.reset = self.custom_train_dataset.reset
                return dataloader
            except Exception as e:
                logger.error(f"Error creating dataloader: {e}")
                raise



