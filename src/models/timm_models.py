import torch
from timm import create_model

from src.models.torch_base_model import TorchBaseModel


class ClassificationTimmModel(TorchBaseModel):
    def __init__(
        self, model_name: str, num_classes: int, pretrained: bool, **kwargs
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.model = create_model(
            model_name=self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
        )

    def inference(self, input: torch.Tensor):
        return self.model(input)


class TimmMobileNetV3(ClassificationTimmModel):
    def __init__(self, num_classes: int, pretrained: bool, **kwargs) -> None:
        super().__init__(
            model_name="mobilenetv3_large_100",
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs,
        )


class TimmMobileViTV2(ClassificationTimmModel):
    def __init__(self, num_classes: int, pretrained: bool, **kwargs) -> None:
        super().__init__(
            model_name="mobilevitv2_100",
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs,
        )
