import torch
from typing import Optional, Tuple, Union, List
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import logging
logger = logging.get_logger(__name__)
import types
from transformers.models.opt.modeling_opt import OPTAttention
from IPython import embed
def opt_attn_shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    return tensor.view(bsz, seq_len, self.num_heads, tensor.shape[-1]//self.num_heads).transpose(1, 2).contiguous()
OPTAttention._shape = opt_attn_shape
def opt_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    def qk_shape(tensor: torch.Tensor, seq_len: int, bsz: int, qk_dim: int):
        return tensor.view(bsz, seq_len, self.num_heads, qk_dim).transpose(1, 2).contiguous()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    qk_dim = query_states.shape[-1] // self.num_heads
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = qk_shape(self.k_proj(hidden_states), -1, bsz, qk_dim)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = qk_shape(self.k_proj(hidden_states), -1, bsz, qk_dim)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    proj_shape_qk = (bsz * self.num_heads, -1, qk_dim)
    query_states = qk_shape(query_states, tgt_len, bsz, qk_dim)
    query_states = query_states.view(*proj_shape_qk)
    key_states = key_states.view(*proj_shape_qk)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
    if attn_weights.dtype == torch.float16:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, past_key_value

OPTAttention.forward = opt_attention_forward

from transformers.models.opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding, OPTDecoderLayer, OPTPreTrainedModel
def decoder__init__(self, config: OPTConfig):
    super(OPTPreTrainedModel, self).__init__(config)
    self.dropout = config.dropout
    self.layerdrop = config.layerdrop
    self.padding_idx = config.pad_token_id
    self.max_target_positions = config.max_position_embeddings
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
    self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

    if config.word_embed_proj_dim != config.hidden_size:
        self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
    else:
        self.project_out = None

    if config.word_embed_proj_dim != config.hidden_size:
        self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
    else:
        self.project_in = None

    # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
    # with checkpoints that have been fine-tuned before transformers v4.20.1
    # see https://github.com/facebookresearch/metaseq/pull/164
    if config.do_layer_norm_before and not config._remove_final_layer_norm:
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
        )
    else:
        self.final_layer_norm = None

    self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)]) # config.num_hidden_layers
    self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    self.gradient_checkpointing = False
    # Initialize weights and apply final processing
    self.post_init()

from transformers.models.opt.modeling_opt import OPTDecoder
OPTDecoder.__init__ = decoder__init__