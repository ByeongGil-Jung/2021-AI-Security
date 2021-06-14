from pathlib import Path
from typing import List, Any, Union, Dict, IO, Optional, Callable
import os
import pickle

import torch
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

from dataset.utils import get_converted_target_list
from domain.criterion import Criterion, CAELoss, CenterLoss
from domain.metadata import ModelMetadata, ModelFileMetadata
from model.base import ModelBase
from model.metrics import roc_auc
from model.occ import OCC
from properties import APPLICATION_PROPERTIES
from logger import logger


class IDConvCAE(ModelBase):

    def __init__(self, model_metadata, *args, **kwargs):  # 117
        super(IDConvCAE, self).__init__(model_metadata=model_metadata, *args, **kwargs)
        self._set_hyperparameters(arg=self.arg)
        self._set_metrics()

        self.num_latent = self.arg.hparams.num_latent

        # Config
        self.model_loss_function = CAELoss(
            num_classes=self.num_classes,
            num_latent=self.num_latent,
            d_in=self.arg.hparams.d_in,
            d_out=self.arg.hparams.d_out,
            lambda_tc=self.arg.hparams.lambda_tc,
            lambda_out=self.arg.hparams.lambda_out,
            lambda_cent=self.arg.hparams.lambda_cent
        )
        self.save_hyperparameters()

        # Layers
        self.encoder_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=3, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(16, 8, 3, stride=2, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=1)
        )
        self.fc_in = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=self.num_latent, bias=True)
        )
        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_latent, out_features=32, bias=True)
        )
        self.decoder_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 16, 3, stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            torch.nn.Tanh()
        )

        # self.ae_parameters = torch.nn.Sequential(self.encoder_1, self.decoder_1)

        # Centers
        # self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.num_classes), requires_grad=True)

        # Init
        self.init_weight(method="he")

    def _set_hyperparameters(self, arg):
        super()._set_hyperparameters(arg=arg)

        self.latent_dir_path = os.path.join(self.model_file_metadata.get_model_outputs_dir_path(), "latent")

        Path(self.latent_dir_path).mkdir(exist_ok=True, parents=True)

    def forward(self, x):
        en_out = self.encoder_1(x)

        en_out_shape = en_out.shape
        z = self.fc_in(en_out.view(en_out.size(0), -1))
        de_in = self.fc_out(z)

        x_hat = self.decoder_1(de_in.reshape(shape=en_out_shape))

        return x_hat, z

    def training_step(self, batch, batch_idx):
        x_id_batch, y_id_batch = batch["train_id"]
        x_ood_batch, y_ood_batch = batch["train_ood"]
        # y_pred_batch, z_in = self(x_batch)  # Hypothesis

        # x_id_batch = x_id_batch.view(x_id_batch.size(0), -1)
        # x_ood_batch = x_ood_batch.view(x_ood_batch.size(0), -1)
        # @TODO noise 추가할 것

        en_id_batch = self.encoder_1(x_id_batch)

        en_id_batch_shape = en_id_batch.shape
        z_in = self.fc_in(en_id_batch.view(en_id_batch.size(0), -1))
        de_id_batch = self.fc_out(z_in)

        x_id_hat_batch = self.decoder_1(de_id_batch.reshape(shape=en_id_batch_shape))

        en_ood_batch = self.encoder_1(x_ood_batch)

        en_ood_batch_shape = en_ood_batch.shape
        z_out = self.fc_in(en_ood_batch.view(en_ood_batch.size(0), -1))
        de_ood_batch = self.fc_out(z_out)

        x_ood_hat_batch = self.decoder_1(de_ood_batch.reshape(shape=en_ood_batch_shape))

        # prob_batch = F.softmax(input=y_pred_batch)
        # loss = F.nll_loss(input=torch.log(prob_batch), target=y_batch)

        """
        Optimize centers
        """

        """
        Optimize model
        """
        loss = self.model_loss_function(x=x_id_batch, x_hat=x_id_hat_batch, target=y_id_batch, z_in=z_in, z_out=None)
        # loss = Criterion.mse_loss(x=x_batch, x_hat=hypothesis_batch)
        # self.manual_backward()

        # Logging
        self.log(APPLICATION_PROPERTIES.TRAIN_LOSS_REPR, loss, on_step=False, on_epoch=True)

        return loss

    def save_best_model_output(self, datamodule=None, version=None):
        if version:
            model_version_dir_path = self.model_file_metadata.get_version_dir_path(version=version)
        else:
            latest_version = self.model_file_metadata.latest_version
            model_version_dir_path = self.model_file_metadata.get_version_dir_path(version=latest_version)

        if not datamodule:
            datamodule = self.datamodule

        model_checkpoints_dir_path = self.model_file_metadata.get_model_checkpoints_dir_path(version=version)
        # model_checkpoints_dir_path = os.path.join(model_version_dir_path, "checkpoints")
        # self.latent_dir_path = os.path.join(model_version_dir_path, "latent")
        latent_dir_path = os.path.join(self.model_file_metadata.get_model_outputs_dir_path(version=version), "latent")
        Path(latent_dir_path).mkdir(parents=True, exist_ok=True)

        for checkpoint_file_name in os.listdir(model_checkpoints_dir_path):
            checkpoint_file_path = os.path.join(model_checkpoints_dir_path, checkpoint_file_name)

            if "epoch" in checkpoint_file_name:
                checkpoint_epoch = checkpoint_file_name.split("-")[0].split("=")[1]
            else:
                checkpoint_epoch = "last"

            logger.info(
                f"Start to load Checkpoint {checkpoint_epoch} | Version : {version} | Path : {checkpoint_file_path}")

            self.load_from_checkpoint(checkpoint_path=checkpoint_file_path)
            val_dataloader = datamodule.val_dataloader()

            y_true_list, hypothesis_list, z_list = self.validation(dataloader=val_dataloader)

            latent_file_path = os.path.join(self.latent_dir_path, f"latent_{checkpoint_epoch}.pkl")

            with open(latent_file_path, "wb") as f:
                pickle.dump(dict(y_true_list=y_true_list, y_pred_list=hypothesis_list, z_list=z_list), f)

            logger.info(f"Complete to save latent space, Checkpoint : {checkpoint_epoch}")

    def validation(self, dataloader):
        y_true_list = list()
        hypothesis_list = list()
        z_list = list()

        for batch_idx, batch in enumerate(dataloader):
            y_batch, hypothesis_batch, z_batch = self.validation_step(batch=batch, batch_idx=batch_idx)

            y_true_list.append(y_batch)
            hypothesis_list.append(hypothesis_batch)
            z_list.append(z_batch)

        y_true_list = torch.cat(y_true_list).detach().cpu().numpy()
        hypothesis_list = torch.cat(hypothesis_list).detach().cpu().numpy()
        z_list = torch.cat(z_list).detach().cpu().numpy()

        return y_true_list, hypothesis_list, z_list

    def test(self, dataloader):
        y_true_list = list()
        hypothesis_list = list()
        z_list = list()

        for batch_idx, batch in enumerate(dataloader):
            y_batch, hypothesis_batch, z_batch = self.test_step(batch=batch, batch_idx=batch_idx)

            y_true_list.append(y_batch)
            hypothesis_list.append(hypothesis_batch)
            z_list.append(z_batch)

        y_true_list = torch.cat(y_true_list).detach().cpu().numpy()
        hypothesis_list = torch.cat(hypothesis_list).detach().cpu().numpy()
        z_list = torch.cat(z_list).detach().cpu().numpy()

        return y_true_list, hypothesis_list, z_list

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        # y_pred_batch, z_in = self(x_batch)  # Hypothesis

        # x_batch = x_batch.view(x_batch.size(0), -1)

        en_batch = self.encoder_1(x_batch)

        en_batch_shape = en_batch.shape
        z = self.fc_in(en_batch.view(en_batch.size(0), -1))
        de_batch = self.fc_out(z)

        x_hat_batch = self.decoder_1(de_batch.reshape(shape=en_batch_shape))

        # prob_batch = F.softmax(input=y_pred_batch)
        # loss = F.nll_loss(input=torch.log(prob_batch), target=y_batch)

        loss = self.model_loss_function(x=x_batch, x_hat=x_hat_batch, target=y_batch, z_in=z, z_out=None)

        # Logging
        self.log(APPLICATION_PROPERTIES.TEST_LOSS_REPR, loss)

        return y_batch, x_hat_batch, z

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        occ = OCC(occ_name=self.arg.hparams.occ)
        datamodule_params = self.trainer.datamodule.arg.hparams

        occ_model, result_dict, classification_report_dict = occ.run_validation(
            representation_model=self,
            version=None,
            epoch=None,
            train_dataloader=self.trainer.datamodule.train_id_dataloader(),
            test_dataloader=self.trainer.datamodule.val_occ_dataloader(),
            id_targets=datamodule_params.train_id_targets,
            ood_targets=datamodule_params.train_ood_targets,
            is_current_representation_model=False
        )

        auc = classification_report_dict["auc"]

        print(f"AUC : {auc}")

        self.log("auc", auc, on_step=False, on_epoch=True)

        if self.arg.hparams.is_latent_save:
            latent_file_path = os.path.join(self.latent_dir_path, f"latent_{self.current_epoch}.pkl")

            with open(latent_file_path, "wb") as f:
                pickle.dump(dict(
                    result_dict=result_dict,
                    report_dict=classification_report_dict
                ), f)

            logger.info("Complete to save latent space")

    def validation_step(self, batch, batch_idx):
        x_id_batch, y_id_batch = batch["val_id"]
        x_ood_batch, y_ood_batch = batch["val_ood"]
        # y_pred_batch, z_in = self(x_batch)  # Hypothesis

        # x_in_batch = x_in_batch.view(x_in_batch.size(0), -1)
        # x_ood_batch = x_ood_batch.view(x_ood_batch.size(0), -1)

        en_id_batch = self.encoder_1(x_id_batch)

        en_id_batch_shape = en_id_batch.shape
        z_in = self.fc_in(en_id_batch.view(en_id_batch.size(0), -1))
        de_id_batch = self.fc_out(z_in)

        x_id_hat_batch = self.decoder_1(de_id_batch.reshape(shape=en_id_batch_shape))

        en_ood_batch = self.encoder_1(x_ood_batch)

        en_ood_batch_shape = en_ood_batch.shape
        z_out = self.fc_in(en_ood_batch.view(en_ood_batch.size(0), -1))
        de_ood_batch = self.fc_out(z_out)

        x_ood_hat_batch = self.decoder_1(de_ood_batch.reshape(shape=en_ood_batch_shape))

        prob_batch = F.softmax(input=x_id_hat_batch)
        # loss = F.nll_loss(input=torch.log(prob_batch), target=y_batch)

        loss = self.model_loss_function(x=x_id_batch, x_hat=x_id_hat_batch, target=y_id_batch, z_in=z_in, z_out=None)
        # loss = Criterion.mse_loss(x=x_batch, x_hat=hypothesis_batch)

        # Logging
        self.log(APPLICATION_PROPERTIES.VAL_LOSS_REPR, loss, on_step=False, on_epoch=True)

        return y_id_batch, x_id_hat_batch, z_in

    def configure_optimizers(self):
        optimizer = self.get_optimizer(optimizer_name=self.hparams.optimizer)
        scheduler = self.get_lr_scheduler(scheduler_name=self.hparams.lr_scheduler, optimizer=optimizer)

        optimizer_output = optimizer

        if scheduler:
            optimizer_output = [optimizer], [scheduler]

        return optimizer_output
