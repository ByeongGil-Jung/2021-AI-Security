import torch
from sklearn.metrics import classification_report
from sklearn.svm import OneClassSVM

from domain.base import Domain
from dataset.utils import get_converted_target_list
from model.metrics import roc_auc
from logger import logger


class OCC(Domain):

    def __init__(self, occ_name="linear_ocsvm", is_input_flatten=False):
        super().__init__()
        self.occ_name = occ_name
        self.is_input_flatten = is_input_flatten
        self.model = self.init(occ_name=occ_name)

    def init(self, occ_name):
        model = None

        # Init OCC
        if occ_name == "linear_ocsvm":
            model = OneClassSVM(kernel="linear", verbose=True)
        if occ_name == "rbf_ocsvm":
            model = OneClassSVM(kernel="rbf", verbose=True)
        if not model:
            raise ValueError("Plz put the occ name, correctly. (ex, occ_name=linear_ocsvm)")

        return model

    def run_vanilla(self, train_dataloader, test_dataloader, id_targets, ood_targets):
        train_x_list = list()
        test_x_list = list()
        test_y_list = list()

        logger.info("Start to load dataset")
        # Train
        for i, (x_batch, y_batch) in enumerate(train_dataloader):
            if self.is_input_flatten:
                x_batch = x_batch.view(x_batch.size(0), -1)

            # hypothesis_list.append(hypothesis_batch)
            train_x_list.append(x_batch)

        train_x_list = torch.cat(train_x_list).detach().cpu().numpy()

        # Test
        for i, (x_batch, y_batch) in enumerate(test_dataloader):
            x_batch = x_batch.view(x_batch.size(0), -1)

            # hypothesis_list.append(hypothesis_batch)
            test_x_list.append(x_batch)
            test_y_list.append(y_batch)

        test_x_list = torch.cat(test_x_list).detach().cpu().numpy()
        test_y_list = torch.cat(test_y_list).detach().cpu().numpy()

        test_y_list = get_converted_target_list(
            y_true_list=test_y_list,
            id_targets=id_targets,
            ood_targets=ood_targets
        )
        logger.info("Complete to load dataset")

        logger.info(f"Start to train the OCC, {self.occ_name}")
        self.model.fit(X=train_x_list)
        logger.info(f"Complete to train the OCC, {self.occ_name}")

        logger.info(f"Start to test the OCC, {self.occ_name}")
        test_y_pred_list = self.model.predict(X=test_x_list)
        score_list = self.model.score_samples(test_x_list)
        logger.info(f"Complete to test the OCC, {self.occ_name}")

        auc = roc_auc(y_true=test_y_list, y_score=score_list)
        classification_report_dict = classification_report(
            y_true=test_y_list,
            y_pred=test_y_pred_list,
            output_dict=True
        )

        classification_report_dict["auc"] = auc

        return classification_report_dict

    def run(self, representation_model, dataloader, id_targets, ood_targets):
        test_y_true_list = list()
        test_z_list = list()

        logger.info("Start to transform validation features to latent")
        for i, (x_batch, y_batch) in enumerate(dataloader):
            if self.is_input_flatten:
                x_batch = x_batch.view(x_batch.size(0), -1)

            hypothesis_batch, z_batch = representation_model.forward(x_batch)

            # hypothesis_list.append(hypothesis_batch)
            test_y_true_list.append(y_batch)
            test_z_list.append(z_batch)

        test_y_true_list = torch.cat(test_y_true_list).detach().cpu().numpy()
        test_z_list = torch.cat(test_z_list).detach().cpu().numpy()

        test_y_true_list = get_converted_target_list(
            y_true_list=test_y_true_list,
            id_targets=id_targets,
            ood_targets=ood_targets
        )
        logger.info("Complete to transform validation features to latent")

        logger.info(f"Start to validate the OCC, {self.occ_name}")
        test_y_pred_list = self.model.predict(X=test_z_list)
        score_list = self.model.score_samples(test_z_list)
        logger.info(f"Complete to validate the OCC, {self.occ_name}")

        auc = roc_auc(y_true=test_y_true_list, y_score=score_list)
        classification_report_dict = classification_report(
            y_true=test_y_true_list,
            y_pred=test_y_pred_list,
            output_dict=True
        )

        classification_report_dict["auc"] = auc

        return classification_report_dict

    def run_validation(self, representation_model, version, epoch, train_dataloader, test_dataloader, id_targets, ood_targets, is_current_representation_model=True):
        # Load Representation model
        if is_current_representation_model:
            representation_model = representation_model.load_checkpoint(epoch=epoch, version=version)
        else:
            representation_model = representation_model
        # representation_model.to("cpu")
        # representation_model.eval()
        # print(representation_model)

        # Train OCC
        y_true_list = list()
        hypothesis_list = list()
        z_list = list()

        logger.info("Start to transform training features to latent")
        for i, (x_batch, y_batch) in enumerate(train_dataloader):
            if self.is_input_flatten:
                x_batch = x_batch.view(x_batch.size(0), -1)

            x_batch = x_batch.to("cuda")
            hypothesis_batch, z_batch = representation_model.forward(x_batch)

            # hypothesis_list.append(hypothesis_batch)
            y_true_list.append(y_batch)
            z_list.append(z_batch)

        # y_true_list = torch.stack(y_true_list, dim=0).detach().cpu().numpy()
        # hypothesis_list = torch.cat(hypothesis_list).detach().numpy()
        z_list = torch.cat(z_list).detach().cpu().numpy()
        logger.info("Complete to transform training features to latent")

        logger.info(f"Start to train the OCC, {self.occ_name}")
        self.model.fit(X=z_list)
        logger.info(f"Complete to train the OCC, {self.occ_name}")

        # Validation OCC
        test_y_true_list = list()
        test_hypothesis_list = list()
        test_z_list = list()

        logger.info("Start to transform validation features to latent")
        for i, (x_batch, y_batch) in enumerate(test_dataloader):
            if self.is_input_flatten:
                x_batch = x_batch.view(x_batch.size(0), -1)

            x_batch = x_batch.to("cuda")
            hypothesis_batch, z_batch = representation_model.forward(x_batch)

            test_hypothesis_list.append(hypothesis_batch)
            test_y_true_list.append(y_batch)
            test_z_list.append(z_batch)

        test_y_true_list = torch.cat(test_y_true_list).detach().cpu().numpy()
        test_hypothesis_list = torch.cat(test_hypothesis_list).detach().cpu().numpy()
        test_z_list = torch.cat(test_z_list).detach().cpu().numpy()

        test_y_true_list = get_converted_target_list(
            y_true_list=test_y_true_list,
            id_targets=id_targets,
            ood_targets=ood_targets
        )
        logger.info("Complete to transform validation features to latent")

        logger.info(f"Start to validate the OCC, {self.occ_name}")
        test_y_pred_list = self.model.predict(X=test_z_list)
        score_list = self.model.score_samples(test_z_list)
        logger.info(f"Complete to validate the OCC, {self.occ_name}")

        auc = roc_auc(y_true=test_y_true_list, y_score=score_list)
        classification_report_dict = classification_report(
            y_true=test_y_true_list,
            y_pred=test_y_pred_list,
            # target_names=[f"class_{i}" for i in range(self.num_classes)],
            output_dict=True
        )

        result_dict = dict(
            y_true_list=test_y_true_list,
            y_pred_list=test_hypothesis_list,
            z_list=test_z_list,
        )

        classification_report_dict["auc"] = auc

        # print(classification_report(
        #     y_true=test_y_true_list,
        #     y_pred=test_y_pred_list,
        #     # target_names=[f"class_{i}" for i in range(self.num_classes)],
        #     output_dict=False
        # ))
        # print(f"AUC : {auc}")

        return self.model, result_dict, classification_report_dict
