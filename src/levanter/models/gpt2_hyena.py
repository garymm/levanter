"""The GPT2 architecture, but with Hyena instead of Attention / Transformer."""

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Type

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.models.attention import AttentionMask, dot_product_attention
from levanter.models.gpt2 import Gpt2Embeddings, Gpt2Mlp
from levanter.models.lm_model import LmConfig, LmHeadModel


@LmConfig.register_subclass("gpt2_hyena")
@dataclass(frozen=True)
class Gpt2HyenaConfig(LmConfig):
    seq_len: int = 1024
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12

    # how much to scale the embedding dim for the mlp layer
    mlp_scale: int = 4

    initializer_range: float = 0.02
    # dropout doesn't really help so we 0 it out by default
    embed_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    hyena_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"

    # mistral tweaks:
    scale_hyena_by_inverse_layer_idx: bool = False
    upcast_hyena: bool = False

    gradient_checkpointing: bool = True  # better to just always use this
    gradient_checkpointing_block_size: int = 5

    use_bias: bool = True

    # Axes
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.hidden_dim * self.mlp_scale))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    @property
    def model_type(self) -> Type["Gpt2HyenaModel"]:
        return Gpt2HyenaModel

    def flops_per_token(self, vocab_size: int) -> Optional[float]:
        # TODO: implement
        return None


class Gpt2Hyena(eqx.Module):
    config: Gpt2HyenaConfig = eqx.field(static=True)

    c_attn: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    c_proj: hnn.Linear  # output projection from [heads, head_dim] -> [embed]
    inference: bool

    @staticmethod
    def init(config: Gpt2HyenaConfig, *, key) -> "Gpt2Hyena":
        Qkv = Axis("qkv", size=3)
        use_bias = config.use_bias
        Embed = config.Embed

        k_c, k_proj = jrandom.split(key, 2)
        c_attn = hnn.Linear.init(
            In=Embed, Out=(Qkv, config.Heads, config.HeadSize), key=k_c, use_bias=use_bias, out_first=False
        )
        c_proj = hnn.Linear.init(
            In=(config.Heads, config.HeadSize), Out=Embed, key=k_proj, use_bias=use_bias, out_first=False
        )

        return Gpt2Hyena(config, c_attn, c_proj, inference=False)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[AttentionMask | NamedArray], layer_idx, *, key):
        k_drop, k_attn, k_out = hax.jax_utils.maybe_rng_split(key, 3)
        qkv_out = self.c_attn(x, key=k_attn).rearrange((..., "qkv", "heads", "position", "head_size"))
        q, k, v = qkv_out.unbind("qkv")

        # Rename k and v's Pos as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # mistral tweak: attention scores can overflow FP16, or just be too imprecise, so upcast to FP32
        if self.config.scale_hyena_by_inverse_layer_idx:
            q = q / (layer_idx + 1.0)

        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask=mask,
            inference=self.inference,
            use_flash=self.config.use_flash_attention,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            prng=k_drop,
            attention_dtype=jnp.float32 if self.config.upcast_hyena else None,
        )

        attn_output = attn_output.astype(x.dtype)
        attn_output = self.c_proj(attn_output, key=k_out)

        return attn_output


class Gpt2HyenaBlock(eqx.Module):
    ln_1: hnn.LayerNorm
    hyena: Gpt2Hyena
    ln_2: hnn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: hnn.Dropout

    @staticmethod
    def init(config: Gpt2HyenaConfig, *, key) -> "Gpt2HyenaBlock":
        k_attn, k_mlp = jrandom.split(key, 2)

        ln_1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        attn = Gpt2Hyena.init(config, key=k_attn)
        ln_2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        mlp = Gpt2Mlp.init(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias)
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)

        return Gpt2HyenaBlock(ln_1, attn, ln_2, mlp, resid_dropout)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[AttentionMask | NamedArray], layer_idx, *, key):
        k1, k2, k3, k4 = haliax.jax_utils.maybe_rng_split(key, 4)

        hyena_output = self.hyena(self.ln_1(x), mask=mask, layer_idx=layer_idx, key=k1)
        hyena_output = self.resid_dropout(hyena_output, key=k2)
        x = x + hyena_output

        ff_output = self.mlp(self.ln_2(x), key=k3)
        ff_output = self.resid_dropout(ff_output, key=k4)
        x = x + ff_output

        return x


class Gpt2HyenaBackbone(ModuleWithStateDictSerialization):
    config: Gpt2HyenaConfig = eqx.field(static=True)
    blocks: Stacked[Gpt2HyenaBlock]
    ln_f: hnn.LayerNorm

    @staticmethod
    def init(config: Gpt2HyenaConfig, *, key):
        # vectorize the blocks
        blocks = Stacked.init(config.Layers, Gpt2HyenaBlock, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return Gpt2HyenaBackbone(config, blocks, ln_f)

    @named_call
    def __call__(self, x: NamedArray, attn_mask: Optional[AttentionMask | NamedArray], *, key=None) -> NamedArray:
        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.blocks.fold(x, attn_mask, hax.arange(self.config.Layers), key=keys)
        x = self.ln_f(x)

        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"blocks": "h"}


class Gpt2HyenaModel(LmHeadModel[Gpt2HyenaConfig]):
    backbone: Gpt2HyenaBackbone
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.backbone.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @classmethod
    def init(cls, Vocab: Axis, config: Gpt2HyenaConfig, *, key) -> "Gpt2HyenaModel":
        k_t, k_embeddings = jrandom.split(key, 2)
        backbone = Gpt2HyenaBackbone.init(config, key=k_t)
        embeddings = Gpt2Embeddings.init(
            Vocab,
            # Our config type has everything it needs, but is not a subclass of Gpt2Config
            config,  # type: ignore
            key=k_embeddings,
        )

        return Gpt2HyenaModel(backbone, embeddings)

    def activations(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        k_embed, k_backbone = haliax.jax_utils.maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids, key=k_embed)
        x = self.backbone(x, attn_mask, key=k_backbone)

        return x

    def get_lm_head(self) -> hax.NamedArray:
        return self.embeddings.token_embeddings.weight

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "Gpt2HyenaModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"backbone": None, "embeddings": None}


ACT2FN: Dict[str, Callable] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": partial(hnn.gelu, approximate=False),
    "gelu_new": partial(hnn.gelu, approximate=True),
    "quick_gelu": hnn.quick_gelu,
}
