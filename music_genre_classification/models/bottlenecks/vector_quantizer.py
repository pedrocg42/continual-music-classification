from typing import Union

import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        codes_per_codebook: int = 4096,
        num_codebooks: int = 256,
        vq_decay: float = 0.95,
        threshold_ema_dead_code: int = 1e-4,
        **kwargs,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codes_per_codebook = codes_per_codebook
        self.num_codebooks = num_codebooks
        self.vq_decay = vq_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # embedding dimension and number of key-value pairs must be divisible by number of codes
        assert (self.embedding_dim % self.num_codebooks) == 0

        self.vector_quantizer = VectorQuantize(
            dim=self.embedding_dim,
            codebook_dim=self.embedding_dim // self.num_codebooks,
            codebook_size=self.codes_per_codebook,
            heads=self.num_codebooks,
            separate_codebook_per_head=True,
            decay=self.vq_decay,
            threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.1·batch-size·h·w/num-pairs)
        )

    def forward(self, embeddings):
        encoder_output_size = embeddings.shape[-1]
        batch_size = embeddings.size()[0]

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
