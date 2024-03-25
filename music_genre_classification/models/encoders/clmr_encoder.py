from collections import OrderedDict
import os
import tempfile
import zipfile
import torch
import torch.nn as nn
from clmr.models.sample_cnn import SampleCNN
import urllib
import config


class ClmrEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.pretrained_weights_path = "https://github.com/Spijkervet/CLMR/releases/download/2.0/clmr_checkpoint_10000.zip"

        # Loading model weights
        self.encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.load_pretrained_weights()

        self.output_size = 512

    def load_pretrained_weights(self) -> None:

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:

            model_path = os.path.join(
                temp_dir, "clmr_checkpoint_10000", "clmr_checkpoint_10000.pt"
            )
            if not os.path.isfile(model_path):
                # Define the path to save the file
                zip_path = os.path.join(temp_dir, "clmr_checkpoint_10000.zip")

                # Download the file
                urllib.request.urlretrieve(self.pretrained_weights_path, zip_path)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(os.path.dirname(model_path))

            # Load the weights
            state_dict = torch.load(
                model_path, map_location=torch.device(config.device)
            )
            state_dict = OrderedDict(
                (k.replace("encoder.", ""), v)
                for k, v in state_dict.items()
                if "projector" not in k
            )
            self.encoder.load_state_dict(state_dict)

    def forward(self, inputs: torch.Tensor):
        outputs = self.encoder.forward_features(inputs)
        return outputs
