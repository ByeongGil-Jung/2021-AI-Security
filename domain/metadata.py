import os
from datetime import datetime
from pathlib import Path

from domain.base import Domain, Hyperparameters
from logger import logger
from properties import APPLICATION_PROPERTIES


class Metadata(Domain):

    def __init__(self, *args, **kwargs):
        super(Metadata, self).__init__(*args, **kwargs)


class Information(Hyperparameters):

    def __init__(self, *args, **kwargs):
        super(Information, self).__init__(*args, **kwargs)


class ModelMetadata(Metadata):

    def __init__(self, model_name, information=None, *args, **kwargs):
        super(ModelMetadata, self).__init__(*args, **kwargs)
        self.model_name = model_name
        self.information = information
        self.model_file_metadata = ModelFileMetadata(model_name=model_name)

    def __repr__(self):
        return f"{self.model_name}"

    # def init(self):
    #     self.model_file_metadata = ModelFileMetadata(model_name=self.model_name)


class ModelFileMetadata(Metadata):

    def __init__(self, model_name, version=None, plot_ext=".png", form="date", *args, **kwargs):
        super(ModelFileMetadata, self).__init__(*args, **kwargs)
        self.form = form
        self.model_name = model_name
        self.current_version = version if version else self.new_version_index
        self.model_dir_path = os.path.join(APPLICATION_PROPERTIES.MODEL_RESULT_DIRECTORY_PATH, self.model_name)
        self.model_latest_version_dir_path = os.path.join(self.model_dir_path, str(self.latest_version))

        # Current version
        self.current_version_dir_path = self.get_version_dir_path(version=self.current_version)
        self.current_version_checkpoints_dir_path = self.get_model_checkpoints_dir_path(version=self.current_version)
        self.current_version_outputs_dir_path = self.get_model_outputs_dir_path(version=self.current_version)
        self.current_optimal_lr_plot_path = os.path.join(self.model_latest_version_dir_path, f"optimal_lr{plot_ext}")  # @TODO remove this

        # New version
        # self.current_version = self.new_version_index
        # self.new_version_dir_path = self.get_version_dir_path(version=self.new_version)
        # self.new_version_checkpoints_dir_path = self.get_model_checkpoints_dir_path(version=self.new_version)

        # Extension
        self.plot_ext = plot_ext

    def __repr__(self):
        return f"{self.model_dir_path}"

    @property
    def latest_version(self):
        if self.get_version_list():
            return sorted(self.get_version_list(), key=lambda version: int(version))[-1]
        else:
            return self.current_version

    def get_version_list(self, is_absolute_path=False):
        version_list = list()
        model_dir_path = self.model_dir_path

        if os.path.isdir(model_dir_path):
            for version in os.listdir(model_dir_path):
                if os.path.isdir(os.path.join(model_dir_path, version)):
                    if is_absolute_path:
                        version_path = os.path.join(model_dir_path, version)
                        version_list.append(version_path)
                    else:
                        version_list.append(version)

        return version_list

    def get_checkpoints_list(self, version=None, is_absolute_path=False):
        version = version if version else self.current_version
        checkpoints_list = list()
        model_checkpoints_dir_path = self.get_model_checkpoints_dir_path(version=version)

        if os.path.isdir(model_checkpoints_dir_path):
            for checkpoint_file_name in os.listdir(model_checkpoints_dir_path):
                if os.path.isfile(os.path.join(model_checkpoints_dir_path, checkpoint_file_name)):
                    if is_absolute_path:
                        checkpoint_file_path = os.path.join(model_checkpoints_dir_path, checkpoint_file_name)
                        checkpoints_list.append(checkpoint_file_path)
                    else:
                        checkpoints_list.append(checkpoint_file_name)

        return checkpoints_list

    def get_version_dir_path(self, version=None):
        version = version if version else self.current_version

        version_dir_path = os.path.join(self.model_dir_path, version)

        return version_dir_path

    def get_model_checkpoints_dir_path(self, version=None):
        version = version if version else self.current_version

        return os.path.join(self.get_version_dir_path(version=version), "checkpoints")

    def get_model_outputs_dir_path(self, version=None):
        version = version if version else self.current_version

        return os.path.join(self.get_version_dir_path(version=version), "outputs")

    @property
    def new_version_index(self):
        version = ""

        if self.form == "date":
            version = datetime.now().strftime("%Y%m%d%H%M")

        return version

    def create_directories(self, is_init=True):
        version = self.current_version

        # Create directories
        version_dir_path = os.path.join(self.model_dir_path, version)
        Path(version_dir_path).mkdir(parents=True, exist_ok=True)

        version_checkpoints_dir_path = self.get_model_checkpoints_dir_path(version=version)
        Path(version_checkpoints_dir_path).mkdir(parents=True, exist_ok=True)

        version_outputs_dir_path = self.get_model_outputs_dir_path(version=version)
        Path(version_outputs_dir_path).mkdir(parents=True, exist_ok=True)

        if is_init:
            self.__init__(model_name=self.model_name, version=version, plot_ext=self.plot_ext)

        logger.debug(f"Create current version {version}, Path : {version_dir_path}")
