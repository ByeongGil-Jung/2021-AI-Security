# from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from domain.base import Hyperparameters
from domain.metadata import ModelMetadata
from properties import APPLICATION_PROPERTIES
from logger import logger
from model.base import ModelBase
from trainer.callback import EarlyStoppingCallback, ModelCheckpointCallback, StatusListenerCallback
from trainer.logging import TensorBoardCustomLogger


class TrainerBase(pl.Trainer):

    def __init__(self, model_metadata: ModelMetadata, model: ModelBase, *args, **kwargs):
        self.model_metadata = model_metadata
        self.model_obj = model
        self.arg = Hyperparameters(kwargs)
        super(TrainerBase, self).__init__(*args, **self._convert_arguments(arg=self.arg))

    def _convert_arguments(self, arg):
        arg = arg.copy()
        model_metadata = self.model_metadata

        # Hyperparameters
        if "hparams" in arg:
            self._set_hyperparameters(hparams=arg.hparams)

            del arg.hparams

        # Callbacks
        if "callbacks" in arg:
            callbacks = arg.callbacks
            callback_list = list()

            # Callback objects
            if "status_listener_callback" in callbacks:
                callback_list.append(StatusListenerCallback()) if callbacks.status_listener_callback else None
            if "model_checkpoint_callback" in callbacks:
                checkpoint_hparam_dict = callbacks.model_checkpoint_callback

                if "dirpath" in checkpoint_hparam_dict and checkpoint_hparam_dict.dirpath == "auto":
                    checkpoint_hparam_dict.dirpath = self.model_metadata.model_file_metadata.current_version_checkpoints_dir_path

                callback_list.append(ModelCheckpointCallback(**callbacks.model_checkpoint_callback))
            if "early_stopping_callback" in callbacks:
                callback_list.append(EarlyStoppingCallback(**callbacks.early_stopping_callback))
            if "others ..." in callbacks:
                pass

            arg.callbacks = callback_list
        else:
            arg.callbacks = [StatusListenerCallback()]

        # Logger
        if "logger" in arg:
            if "tensor_board_logger" == arg.logger:
                arg.logger = TensorBoardCustomLogger(model_metadata=model_metadata)
            if "others ..." == arg.logger:
                pass

        # Auto lr find
        if "is_auto_lr_find" in arg:
            self.is_auto_lr_find = arg.is_auto_lr_find

            del arg.is_auto_lr_find

        return arg

    def on_fit_start(self):
        # Generate directories
        self.model_metadata.model_file_metadata.create_directories()
        # self.model_obj.save_hyperparameters()
        # self.model.save_hyperparameters)
        # self.model.save_hyperparameters()
        super().on_fit_start()

    def _set_hyperparameters(self, hparams):
        self.hparams = hparams

    def lr_find(self, model, train_loader, val_loader):
        logger.info(f"Start to find the optimal learning rate ...")
        self.lr_finder = self.tuner.lr_find(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

        optimal_lr = self.lr_finder.suggestion()
        model.hparams.lr = optimal_lr
        logger.info(f"Finished finding the optimal learning rate : {optimal_lr}")
