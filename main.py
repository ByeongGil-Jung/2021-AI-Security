import argparse

from config.factory import HyperparameterFactory
from dataset.factory import DataModuleFactory
from domain.metadata import ModelMetadata
from logger import logger
from model.factory import ModelFactory
from properties import APPLICATION_PROPERTIES
from trainer.base import TrainerBase


def main(args):
    model_name = args.model.lower()
    data_name = args.data.lower()
    stage = args.stage.lower()
    tqdm_env = args.tqdm_env.lower()

    model_metadata = ModelMetadata(model_name=model_name, information=None)

    # Arguments controller
    hyperparameter_factory = HyperparameterFactory.create(data_name=data_name, model_name=model_name)
    datamodule_params = hyperparameter_factory.datamodule_params
    trainer_params = hyperparameter_factory.trainer_params
    model_params = hyperparameter_factory.model_params

    # DataModule controller
    datamodule = DataModuleFactory.create(data_name=data_name)
    datamodule = datamodule(**datamodule_params)

    datamodule.prepare_data()
    datamodule.setup(stage=stage)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Trainer controller
    trainer = TrainerBase(model_metadata=model_metadata, **trainer_params)

    # Model controller
    model = ModelFactory.create(model_name=model_name, model_metadata=model_metadata, model_params=model_params)

    # Training & Validation
    if stage == "fit":
        trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # Testing
    if stage == "test":
        trainer.test(model=model, test_dataloaders=test_loader)

    logger.info(f"Finished all processes, result directory : {model_metadata.model_file_metadata.model_latest_version_dir_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pytorch Lightning Project Template [Byeonggil Jung (Korea Univ, AIR Lab)]")
    parser.add_argument("--model", required=False, default="FC", help="Model name")
    parser.add_argument("--data", required=False, default="MNIST", help="Dataset name")
    parser.add_argument("--stage", required=False, default="whole", help="Select the stage, fit | test | whole")
    parser.add_argument("--tqdm_env", required=False, default="script", help="Select the tqdm environment, script | jupyter")
    args = parser.parse_args()

    logger.info(f"Selected parameters : {args}")

    main(args=args)
