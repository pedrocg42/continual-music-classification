from torch import nn, Tensor


class MertClassificationDecoder(nn.Module):
    def __init__(
        self,
        conv1_dict: dict[str, int],
        in_features: int,
        num_classes: int,
        hidden_units: int = 512,
        dropout: float | None = 0.25,
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=conv1_dict["in_channels"],
            out_channels=conv1_dict["out_channels"],
            kernel_size=conv1_dict["kernel_size"],
        )
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=dropout)
        self.hidden_fc = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.normalization = nn.BatchNorm1d(num_features=hidden_units)
        self.activation = nn.ReLU()

        self.fc = nn.Linear(in_features=hidden_units, out_features=num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.conv1d(inputs)
        outputs = self.flatten(outputs)

        outputs = self.dropout(outputs)
        outputs = self.hidden_fc(outputs)
        outputs = self.normalization(outputs)
        outputs = self.activation(outputs)

        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs

    def forward_features(self, inputs: Tensor) -> Tensor:
        outputs = self.conv1d(inputs)
        outputs = self.flatten(outputs)
        return outputs
