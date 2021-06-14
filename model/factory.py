from model.cae_conv import ConvCAE
from model.cae_conv_id import IDConvCAE
from model.cae_conv_mse import ConvCAEMse
from domain.base import Factory


class ModelFactory(Factory):

    def __init__(self):
        super(ModelFactory, self).__init__()

    @classmethod
    def create(cls, model_name, model_metadata, model_params):
        model = None

        if model_name == "conv_cae":
            model = ConvCAE
        if model_name == "conv_cae_id":
            model = IDConvCAE
        if model_name == "conv_cae_mse":
            model = ConvCAEMse

        return model(model_metadata, **model_params)
