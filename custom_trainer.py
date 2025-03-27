from functools import partial
from ultralytics.models.yolo.detect.train import DetectionTrainer
from torch.utils.data import DataLoader
import torch


from ultralytics.models.yolo.detect.train import DetectionTrainer
from torch.utils.data import DataLoader
from Custom_Dataloader import Custom_Dataloader, collate_fn  

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
   
)

logger = logging.getLogger(__name__)

class CustomTrainer(DetectionTrainer):
    def __init__(self, *args, custom_train_dataset=None, **kwargs):
        logger.info("CustomTrainer initialization started")
        super().__init__(*args, **kwargs)
        self.custom_train_dataset = custom_train_dataset
        self.plot_training_labels = lambda *a, **kw: None
        logger.info(f"CustomTrainer initialized with dataset: {custom_train_dataset is not None}")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        logger.info(f"get_dataloader called with mode={mode}")
        if mode == "train" and self.custom_train_dataset is not None:
            logger.info("Using custom dataset in get_dataloader")
            return self.build_dataloader(self.custom_train_dataset, mode=mode, batch=batch_size)
        logger.info("Falling back to parent get_dataloader")
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

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
                )
                logger.info(f"Successfully created dataloader with {len(dataloader)} batches")
                return dataloader
            except Exception as e:
                logger.error(f"Error creating dataloader: {e}")
                raise

    def train(self):
        logger.info("Training started")
        return super().train()


