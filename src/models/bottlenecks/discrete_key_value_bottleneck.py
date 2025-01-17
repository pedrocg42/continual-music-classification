import torch
import torch.nn as nn
from discrete_key_value_bottleneck_pytorch import DiscreteKeyValueBottleneck


class DKVB(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        projection_embedding_dim: int | None = 128,
        dim_memory: int = 10,
        codes_per_codebook: int = 4096,
        num_codebooks: int = 256,
        vq_decay: float = 0.95,
        threshold_ema_dead_code: int = 1e-4,
        value_dimension: int | str = "same",
        **kwargs,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codes_per_codebook = codes_per_codebook
        self.num_codebooks = num_codebooks
        self.value_dimension = value_dimension
        self.vq_decay = vq_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # embedding dimension and number of key-value pairs must be divisible by number of codes
        assert (self.embedding_dim % self.num_codebooks) == 0

        self.dkvb = DiscreteKeyValueBottleneck(
            dim_embed=self.embedding_dim,  # input dimension
            dim=projection_embedding_dim,  # dimension of the projected keys
            codebook_dim=projection_embedding_dim,  # dimension of the input codes of the codebook of VQ
            num_memory_codebooks=self.num_codebooks,  # number of memory codebook
            num_memories=self.codes_per_codebook,  # number of memories per codebook
            dim_memory=dim_memory,  # dimension of the output memories
            decay=self.vq_decay,  # the exponential moving average decay, lower means the keys will change faster
            threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.1·batch-size·h·w/num-pairs)
            average_pool_memories=True,  # whether to average pool the memories
        )

    def forward(self, embeddings: torch.Tensor):
        encoder_output_size = embeddings.shape[-1]
        batch_size = embeddings.size()[0]

        four_dim_features = len(embeddings.size()) == 4

        if four_dim_features:
            embeddings = torch.reshape(
                embeddings,
                (embeddings.shape[0], self.embedding_dim, encoder_output_size**2),
            )  # B, Dim, H, W -> B, Dim, N
            embeddings = torch.permute(embeddings, (0, 2, 1))  # B, Dim, N -> B, N, Dim

        memories = self.dkvb(embeddings)

        if four_dim_features:
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
