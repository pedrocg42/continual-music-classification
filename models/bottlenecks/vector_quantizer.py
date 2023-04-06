from typing import Union

import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 1024,
        codebook_size: int = 8192,
        num_codebooks: int = 1,
        vq_decay: float = 0.99,
        threshold_ema_dead_code: int = 2,
        value_dimension: Union[int, str] = "same",
        **kwargs,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.value_dimension = value_dimension
        self.vq_decay = vq_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # embedding dimension and number of key-value pairs must be divisible by number of codes
        assert (self.embedding_dim % self.num_codebooks) == 0
        assert (self.codebook_size & self.num_codebooks) == 0

        self.vector_quantizer = VectorQuantize(
            dim=self.embedding_dim,
            codebook_dim=self.embedding_dim // self.num_codebooks,
            codebook_size=self.codebook_size,
            heads=self.num_codebooks,
            separate_codebook_per_head=True,
            decay=self.vq_decay,
            threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.8·batch-size·h·w·mz/num-pairs)
        )

    def forward(self, x):
        encoder_output_size = embeddings.shape[-1]
        batch_size = x.size()[0]

        embeddings = torch.reshape(
            embeddings,
            (embeddings.shape[0], self.embedding_dim, encoder_output_size**2),
        )  # B, Dim, H, W -> B, Dim, N
        embeddings = torch.permute(embeddings, (0, 2, 1))  # B, Dim, N -> B, N, Dim

        quantized, _, _ = self.vector_quantizer(
            embeddings
        )  # quantized, indices, commitment loss

        quantized = torch.permute(quantized, (0, 2, 1))  # B, N, Dim -> B, Dim, N
        quantized = torch.reshape(
            quantized,
            (
                batch_size,
                self.embedding_dim,
                encoder_output_size,
                encoder_output_size,
            ),
        )  # B, Dim, N -> B, Dim, H, W

        return quantized
