import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

from domain.base import Domain


class CAELoss(nn.Module):

    def __init__(self, num_classes, num_latent, d_in=0.1, d_out=1, lambda_tc=1, lambda_out=1, lambda_cent=1):
        super(CAELoss, self).__init__()
        self.num_classes = num_classes
        self.num_latent = num_latent
        self.center_arr = nn.Parameter(torch.randn(num_classes, num_latent), requires_grad=True)

        # Hyperparams
        self.d_in = d_in
        self.d_out = d_out
        self.lambda_tc = lambda_tc
        self.lambda_out = lambda_out
        self.lambda_cent = lambda_cent

    def triplet_center_loss(self, z_in_batch, y_batch):
        loss_list = list()

        for z_in, y in zip(z_in_batch, y_batch):
            mask = torch.zeros(self.num_classes, dtype=torch.bool)
            mask[y] = True
            pos_mask, neg_mask = mask, ~mask

            pos_distance = torch.linalg.norm(z_in - self.center_arr[pos_mask])
            neg_distance = torch.as_tensor([torch.linalg.norm(z_in - neg_centroid) for neg_centroid in self.center_arr[neg_mask]]).min()

            loss_list.append(torch.clamp(pos_distance + self.d_in - neg_distance, min=0))

        loss = torch.as_tensor(loss_list).mean()

        return loss

    def outlier_loss(self, z_out_batch):
        loss_list = list()

        for z_out in z_out_batch:
            z_out_distance = torch.linalg.norm(z_out)

            loss_list.append((torch.clamp(self.d_out - z_out_distance, min=0)))

        loss = torch.as_tensor(loss_list).mean()

        return loss

    def forward(self, x, x_hat, target, z_in, z_out=None):
        # Clipping center vector to unit vector
        with torch.no_grad():
            for idx, center in enumerate(self.center_arr):
                self.center_arr[idx] = center / torch.norm(center)

        loss_mse = Criterion.mse_loss(x=x, x_hat=x_hat)
        loss_triplet_center = self.triplet_center_loss(z_in_batch=z_in, y_batch=target)

        loss_outlier = self.outlier_loss(z_out_batch=z_out) if z_out is not None else 0

        # Centroid loss
        center_matmul_mtx = torch.matmul(self.center_arr, self.center_arr.t())
        centroid_orthogonal_loss = torch.linalg.norm(center_matmul_mtx - torch.eye(center_matmul_mtx.shape[0], device="cuda", requires_grad=False))

        loss_total = loss_mse + (self.lambda_tc * loss_triplet_center) + (self.lambda_out * loss_outlier) + (self.lambda_cent * centroid_orthogonal_loss)

        return loss_total


class Criterion(Domain):

    def __init__(self, *args, **kwargs):
        super(Criterion, self).__init__(*args, **kwargs)

    @classmethod
    def mse_loss(cls, x, x_hat):
        return torch.mean(torch.pow((x - x_hat), 2))

    @classmethod
    def reconstruction_error(cls, x, x_hat):
        return ((x_hat - x) ** 2).mean(axis=1)

    @classmethod
    def sae_loss(cls, x, x_hat, latent_data, lambda_):
        return torch.mean(torch.pow((x - x_hat), 2)) + lambda_ * torch.mean(torch.pow(latent_data, 2))
