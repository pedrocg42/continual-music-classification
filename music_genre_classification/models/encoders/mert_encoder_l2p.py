from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import einsum, nn
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutput
from vector_quantize_pytorch import VectorQuantize


class MertEncoderL2P(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        prompt_pool_size: int = 10,  # M
        prompt_length: int = 5,  # L_p
        selection_size: int = 5,  # N
    ) -> None:
        super().__init__()

        self.pretrained = pretrained

        # Loading model weights
        self.encoder = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self.output_size = 768
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Prompt pool
        self.prompt_pool_size = prompt_pool_size
        self.prompt_length = prompt_length
        self.selection_size = selection_size
        self.prompt_pool = PromptPool(
            length=self.prompt_length,
            pool_size=self.prompt_pool_size,
            top_k=self.selection_size,
            embedding_dim=self.output_size,
        )
        self.num_prompts = self.selection_size * self.prompt_length

    def forward(self, inputs: torch.Tensor):
        query = self.query(inputs)
        prompt, key_loss = self.prompt_pool(query)
        prompt = prompt.view(prompt.shape[0], -1, self.output_size)
        outputs = self.forward_encoder(
            **inputs, prompt=prompt, output_hidden_states=True
        )
        all_layer_hidden_states = torch.stack(outputs.hidden_states).permute(
            (1, 0, 2, 3)
        )  # C, B, S, H -> B, C, S, H
        prompt_hidden_states = all_layer_hidden_states[:, :, : self.num_prompts, :]
        outputs = torch.mean(prompt_hidden_states, dim=-2)  # B, C, S, H -> B, C, H
        return outputs, key_loss

    @torch.no_grad()
    def query(self, inputs: torch.Tensor):
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=-2)

    def forward_encoder(
        self,
        input_values: torch.Tensor,
        prompt: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.encoder.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.encoder.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.encoder.config.use_return_dict
        )

        extract_features = self.encoder.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        # add additional cqt features for transformer input
        if self.encoder.config.feature_extractor_cqt:
            features_cqt = self.encoder.feature_extractor_cqt(input_values).transpose(
                1, 2
            )
            features_cqt = features_cqt[
                :, : extract_features.shape[1], :
            ]  # align shape
            extract_features = torch.cat([extract_features, features_cqt], 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.encoder._get_feature_vector_attention_mask(
                extract_features.shape[1] + prompt.shape[1], attention_mask
            )

        hidden_states = self.encoder.feature_projection(extract_features)

        hidden_states = torch.cat([prompt, hidden_states], dim=1)
        hidden_states = self.encoder._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices
        )

        encoder_outputs = self.encoder.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]  # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


class PromptPool(nn.Module):
    def __init__(
        self,
        length: int,  # L_p
        pool_size: int = None,  # M
        top_k: int = None,  # N
        embedding_dim: int = 768,
    ):
        super().__init__()

        self.length = length
        self.pool_size = pool_size
        self.top_k = top_k

        # Probability distribution over the prompt pool
        self.h_sum = torch.zeros(self.pool_size)
        self.num_searches = 0
        self.h = torch.zeros(pool_size)

        self.prompt_keys = nn.Parameter(uniform_init(self.pool_size, embedding_dim))
        self.prompt_values = nn.Parameter(
            uniform_init(self.pool_size, self.length, embedding_dim)
        )

    def forward(self, query: torch.Tensor) -> tuple[torch.Tensor]:
        prompt_key_norm = F.normalize(self.prompt_keys, dim=-1)
        query_norm = F.normalize(query, dim=-1)

        distance = query_norm @ prompt_key_norm.T  # bs, pool_size
        (distance_top_k, distance_top_k_idx) = torch.topk(distance, self.top_k)

        one_hot_idx = F.one_hot(
            distance_top_k_idx, self.pool_size
        ).float()  # bs, top_k, pool_size

        quantized_values = einsum(
            "b n s, s l d -> b n l d", one_hot_idx, self.prompt_values
        )

        self.h_sum += one_hot_idx.sum(axis=0).sum(axis=0).detach().cpu()
        self.num_searches += query.shape[0] * self.top_k
        self.h = self.h_sum / self.num_searches

        # Put pull_constraint loss calculation inside
        key_loss = torch.sum(torch.abs(distance_top_k)) / query.shape[0]

        return (quantized_values, key_loss)
