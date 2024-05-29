from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.bert import modeling_bert

from ane_transformers.reference.layer_norm import LayerNormANE

# Note: Original implementation of distilbert uses an epsilon value of 1e-12
# which is not friendly with the float16 precision that ANE uses by default
EPS = 1e-7


# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    state_dict[prefix + "bias"] = (
        state_dict[prefix + "bias"] / state_dict[prefix + "weight"]
    )
    return state_dict


def linear_to_conv2d_map(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights"""
    for k in state_dict:
        is_internal_proj = all(substr in k for substr in [".layer", ".weight"])
        is_pooler = "pooler" in k
        if is_internal_proj or is_pooler:
            if len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]


class LayerNormANE(LayerNormANE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)


class Embeddings(modeling_bert.BertEmbeddings):
    """Embeddings module optimized for Apple Neural Engine"""

    def __init__(self, config):
        super().__init__(config)
        setattr(self, "LayerNorm", LayerNormANE(config.hidden_size, eps=EPS))


class MultiHeadSelfAttention(modeling_bert.BertSelfAttention):
    """MultiHeadSelfAttention module optimized for Apple Neural Engine"""

    def __init__(self, config):
        super().__init__(config)
        setattr(
            self,
            "query",
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=self.all_head_size,
                kernel_size=1,
            ),
        )

        setattr(
            self,
            "key",
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=self.all_head_size,
                kernel_size=1,
            ),
        )

        setattr(
            self,
            "value",
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=self.all_head_size,
                kernel_size=1,
            ),
        )

    def prune_heads(self, heads):
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        # def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, dim, 1, seq_length)
            key: torch.tensor(bs, dim, 1, seq_length)
            value: torch.tensor(bs, dim, 1, seq_length)
            mask: torch.tensor(bs, seq_length) or torch.tensor(bs, seq_length, 1, 1)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            dim, 1, seq_length) Contextualized layer. Optional: only if `output_attentions=True`
        """
        # Parse tensor shapes for source and target sequences
        assert len(hidden_states.size()) == 4

        bs, dim, dummy, seqlen = hidden_states.size()
        # TODO : check assertions here
        # assert seqlen == key.size(3) and seqlen == value.size(3)
        # assert dim == self.dim
        # assert dummy == 1

        # Project q, k and v
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Validate mask
        if head_mask is not None:
            expected_mask_shape = [bs, seqlen, 1, 1]
            if head_mask.dtype == torch.bool:
                head_mask = head_mask.logical_not().float() * -1e4
            elif head_mask.dtype == torch.int64:
                head_mask = (1 - head_mask).float() * -1e4
            elif head_mask.dtype != torch.float32:
                raise TypeError(f"Unexpected dtype for mask: {head_mask.dtype}")

            if len(head_mask.size()) == 2:
                head_mask = head_mask.unsqueeze(2).unsqueeze(2)

            if list(head_mask.size()) != expected_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `mask` (Expected {expected_mask_shape}, got {list(head_mask.size())}"
                )

        if head_mask is not None:
            raise NotImplementedError

        # Compute scaled dot-product attention
        mh_q = q.split(
            self.attention_head_size, dim=1
        )  # (bs, dim_per_head, 1, max_seq_length) * n_heads
        mh_k = k.transpose(1, 3).split(
            self.attention_head_size, dim=3
        )  # (bs, max_seq_length, 1, dim_per_head) * n_heads
        mh_v = v.split(
            self.attention_head_size, dim=1
        )  # (bs, dim_per_head, 1, max_seq_length) * n_heads

        normalize_factor = float(self.attention_head_size) ** -0.5
        attn_weights = [
            torch.einsum("bchq,bkhc->bkhq", [qi, ki]) * normalize_factor
            for qi, ki in zip(mh_q, mh_k)
        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads

        if head_mask is not None:
            for head_idx in range(self.num_attention_heads):
                attn_weights[head_idx] = attn_weights[head_idx] + head_mask

        attn_weights = [
            aw.softmax(dim=1) for aw in attn_weights
        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads
        attn = [
            torch.einsum("bkhq,bchk->bchq", wi, vi)
            for wi, vi in zip(attn_weights, mh_v)
        ]  # (bs, dim_per_head, 1, max_seq_length) * n_heads

        attn = torch.cat(attn, dim=1)  # (bs, dim, 1, max_seq_length)

        if output_attentions:
            return attn, attn_weights.cat(dim=2)
        else:
            return (attn,)


class SelfOutput(modeling_bert.BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        setattr(
            self,
            "dense",
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=1,
            ),
        )
        setattr(self, "LayerNorm", LayerNormANE(config.hidden_size, eps=EPS))


class AttentionBlock(modeling_bert.BertAttention):
    def __init__(self, config):
        super().__init__(config)
        setattr(self, "self", MultiHeadSelfAttention(config))
        setattr(self, "output", SelfOutput(config))


class Output(modeling_bert.BertOutput):
    def __init__(self, config):
        super().__init__(config)
        setattr(
            self,
            "dense",
            nn.Conv2d(
                in_channels=config.intermediate_size,
                out_channels=config.hidden_size,
                kernel_size=1,
            ),
        )
        setattr(self, "LayerNorm", LayerNormANE(config.hidden_size, eps=EPS))


class Intermediate(modeling_bert.BertIntermediate):
    def __init__(self, config):
        super().__init__(config)
        setattr(
            self,
            "dense",
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.intermediate_size,
                kernel_size=1,
            ),
        )


class Layer(modeling_bert.BertLayer):
    def __init__(self, config):
        super().__init__(config)
        setattr(self, "attention", AttentionBlock(config))
        setattr(self, "intermediate", Intermediate(config))
        setattr(self, "output", Output(config))


class Encoder(modeling_bert.BertEncoder):

    def __init__(self, config):
        super().__init__(config)
        setattr(
            self,
            "layer",
            nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)]),
        )


class Pooler(modeling_bert.BertPooler):
    def __init__(self, config):
        super().__init__(config)
        setattr(
            self,
            "dense",
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=1,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (bs, dim, 1, seq_len) - > (bs,seq_len,dim)
        hidden_states = hidden_states.squeeze(2).transpose(1, 2)
        first_token_tensor = hidden_states[:, 0]
        # TODO: maybe skip recasting to 4D tensor for ANE here ? 
        pooled_output = self.dense(first_token_tensor[:, :, None, None])
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(modeling_bert.BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        setattr(self, "embeddings", Embeddings(config))
        setattr(self, "encoder", Encoder(config))
        setattr(self, "pooler", Pooler(config) if add_pooling_layer else None)
        # TODO: add poooler ??
        # Register hook for unsqueezing nn.Linear parameters to match nn.Conv2d parameter spec
        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError
