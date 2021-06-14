import os

from torch import nn
from torch import optim
import pytorch_lightning as pl
import torchmetrics

from domain.base import Hyperparameters
from domain.metadata import ModelMetadata, ModelFileMetadata
from logger import logger


class ModelBase(pl.LightningModule):

    def __init__(self, model_metadata, *args, **kwargs):
        self.arg = Hyperparameters(kwargs)
        self.model_metadata = model_metadata
        self.model_file_metadata: ModelFileMetadata = self.model_metadata.model_file_metadata
        super(ModelBase, self).__init__()

    def _set_hyperparameters(self, arg):
        arg = arg.copy()

        hparams = arg.hparams

        self.num_classes = hparams.num_classes if "num_classes" in hparams else None
        self.input_size = hparams.input_size if "input_size" in hparams else None

    def _set_metrics(self):
        num_classes = self.num_classes

        # Train
        self.train_acc = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision()
        self.train_recall = torchmetrics.Recall()
        self.train_f1 = torchmetrics.F1(num_classes=num_classes) if num_classes else None
        self.train_auc = torchmetrics.AUROC(num_classes=num_classes) if num_classes else None

        # Validation
        self.validation_acc = torchmetrics.Accuracy()
        self.validation_precision = torchmetrics.Precision()
        self.validation_recall = torchmetrics.Recall()
        self.validation_f1 = torchmetrics.F1(num_classes=num_classes) if num_classes else None
        self.validation_auc = torchmetrics.AUROC(num_classes=num_classes) if num_classes else None

        # Test
        self.test_acc = torchmetrics.Accuracy()
        self.test_precision = torchmetrics.Precision()
        self.test_recall = torchmetrics.Recall()
        self.test_f1 = torchmetrics.F1(num_classes=num_classes) if num_classes else None
        self.test_auc = torchmetrics.AUROC(num_classes=num_classes) if num_classes else None

        # Add ...

    def init_weight(self, method="he"):
        if method == "he":
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight.data)
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight.data)
                elif isinstance(module, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(module.weight.data)
        if method == "xavier":
            pass

    def load_checkpoint(self, epoch, version=None):
        version = version if version else self.model_file_metadata.current_version
        checkpoints_dir_path = self.model_file_metadata.get_model_checkpoints_dir_path(version=version)
        checkpoint_file_path = ""

        for checkpoint_file_name in os.listdir(checkpoints_dir_path):
            if f"epoch={epoch}" in checkpoint_file_name:
                checkpoint_file_path = os.path.join(checkpoints_dir_path, checkpoint_file_name)

        if checkpoint_file_path == "":
            checkpoint_file_path = os.path.join(checkpoints_dir_path, "last.ckpt")

        loaded_model = self.load_from_checkpoint(checkpoint_path=checkpoint_file_path)

        logger.info(f"Complete to load checkpoint, Version : {version}, Epoch : {epoch}")

        return loaded_model

    def get_optimizer(self, optimizer_name):
        optimizer = None

        if optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        # Add ...

        return optimizer

    def get_lr_scheduler(self, scheduler_name, optimizer):
        scheduler = None

        if scheduler_name == "cosine_anealing_warm_restart":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.T_0)
        # Add ...

        return scheduler
