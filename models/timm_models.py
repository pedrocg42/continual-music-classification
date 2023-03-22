from timm import create_model
import torch.nn as nn
import torch


class ClassificationTimmModel(nn.Module):
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

    def forward(self, x: torch.Tensor):
        return self.model(x)


class TimmMobileNetV3(ClassificationTimmModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_name="mobilenetv3_large_100", **kwargs)


class TimmMobileViTV2(ClassificationTimmModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_name="mobilevitv2_100", **kwargs)
