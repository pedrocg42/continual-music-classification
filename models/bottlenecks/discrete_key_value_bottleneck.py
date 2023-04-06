from typing import Union

import torch
import torch.nn as nn
from discrete_key_value_bottleneck_pytorch import DiscreteKeyValueBottleneck


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

        self.key_value_bottleneck = DiscreteKeyValueBottleneck(
            dim=self.embedding_dim,  # input dimension
            codebook_dim=self.embedding_dim // self.num_codebooks,
            num_memory_codebooks=self.num_codebooks,  # number of memory codebook
            num_memories=self.codebook_size,  # number of memories
            dim_memory=self.embedding_dim
            // self.num_codebooks,  # dimension of the output memories
            decay=self.vq_decay,  # the exponential moving average decay, lower means the keys will change faster
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

        memories = self.key_value_bottleneck(embeddings)

        memories = torch.permute(memories, (0, 2, 1))  # B, N, Dim -> B, Dim, N
        memories = torch.reshape(
            memories,
            (
                batch_size,
                self.embedding_dim,
                encoder_output_size,
                encoder_output_size,
            ),
        )  # B, Dim, N -> B, Dim, H, W

        return memories
