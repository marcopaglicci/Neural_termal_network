from functools import partial
from ultralytics.models.yolo.detect.train import DetectionTrainer
from torch.utils.data import DataLoader
from data_loader import YOLODataset, collate_fn 


from ultralytics.models.yolo.detect.train import DetectionTrainer
from torch.utils.data import DataLoader
from data_loader import YOLODataset, collate_fn  # ✅ import corretto

class CustomTrainer(DetectionTrainer):
    def __init__(self, *args, custom_train_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_train_dataset = custom_train_dataset

    def build_dataloader(self, dataset, mode="train", batch=None):
        if mode == "train" and self.custom_train_dataset is not None:
            return DataLoader(
                self.custom_train_dataset,
                batch_size=self.args.batch,
                shuffle=True,
                num_workers=self.args.workers,
                collate_fn=collate_fn,
            )
        return super().build_dataloader(dataset, mode, batch)