import timm
from torch import nn

from components.config import Model


def get_model(
    model_name: Model = Model.NAME,
    pretrained: bool = True,
    global_pool: str = "avg",
) -> nn.Module:
    """Get model function.

    Args:
    ----
        model_name (Model, optional): Model name. Defaults to Model.NAME.
        pretrained (bool, optional): Whether or not to pretrain. Defaults to True.
        global_pool (str, optional): The module name of global pool. Defaults to "avg".

    Returns:
    -------
        nn.Module: The target model

    """
    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        global_pool=global_pool,
        num_classes=0,
    )

    model.eval()

    return model
