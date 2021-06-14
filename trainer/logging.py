from datetime import datetime

from pytorch_lightning.loggers import TensorBoardLogger

from domain.metadata import ModelMetadata
from properties import APPLICATION_PROPERTIES


class TensorBoardCustomLogger(TensorBoardLogger):

    def __init__(self, model_metadata: ModelMetadata, *args, **kwargs):
        self._convert_arguments(model_metadata=model_metadata, kwargs=kwargs)
        super(TensorBoardCustomLogger, self).__init__(*args, **kwargs)

    def _convert_arguments(self, model_metadata, kwargs):
        model_metadata = model_metadata
        model_file_metadata = model_metadata.model_file_metadata
        logger_hparams = kwargs

        logger_hparams["save_dir"] = APPLICATION_PROPERTIES.MODEL_RESULT_DIRECTORY_PATH
        logger_hparams["name"] = model_file_metadata.model_name
        logger_hparams["version"] = model_file_metadata.current_version
