from torch import Tensor, nn


class ClmrClassificationDecoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_units: int = 512,
        dropout: float | None = 0.25,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.hidden_fc = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.normalization = nn.BatchNorm1d(num_features=hidden_units)
        self.activation = nn.ReLU()

        self.fc = nn.Linear(in_features=hidden_units, out_features=num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.dropout(inputs)
        outputs = self.hidden_fc(outputs)
        outputs = self.normalization(outputs)
        outputs = self.activation(outputs)

        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs

    def forward_features(self, inputs: Tensor) -> Tensor:
        return inputs
