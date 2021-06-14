from abc import abstractmethod
from argparse import Namespace
from idlelib.config import _warn
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict, rank_zero_only

from domain.base import Hyperparameters
from properties import APPLICATION_PROPERTIES


rank_zero_warn = rank_zero_only(_warn)


class DataModuleBase(pl.LightningDataModule):

    def __init__(self, *args, **kwargs):
        self.arg = Hyperparameters(kwargs)
        self._convert_arguments(arg=self.arg)
        super(DataModuleBase, self).__init__()

        # Dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        pass

    def _convert_arguments(self, arg):
        # Args
        if arg.data_dir == "auto":
            self.data_dir = APPLICATION_PROPERTIES.DATA_DIRECTORY_PATH

        # Datamodule params
        if "datamodule_params" in arg:
            self.datamodule_params = arg.datamodule_params

        # Hyperparameters
        if "hparams" in arg:
            self.hparams = arg.hparams

        return arg
