"""An implementation of Hyena, a convolutional language model architecture.

Paper: [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)
Official implementation in PyTorch:
 - Overall architecture:
   https://github.com/HazyResearch/safari/blob/fc2d8b18be36bce427a4b9b8073e508c86c8f7ee/src/models/sequence/long_conv_lm.py
 - Hyena operator:
   https://github.com/HazyResearch/safari/blob/541902aca88cb11af4d816ac762f3303e4ff8eea/src/models/sequence/hyena.py
"""

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Type

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, LmWithHfSerializationMixin
from levanter.models.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag


@LmConfig.register_subclass("hyena")
@dataclass(frozen=True)
class HyenaConfig(LmConfig):
    seq_len: int = 1024
    hidden_dim: int = 768
    num_layers: int = 12

    # hyena specific parameters
    order: int = 2  # depth of the Hyena recurrence
    filter_order: int = 64  # width of the FFN parametrizing the implicit filter
    inner_factor: int = 1  # inner dimension multiplier
    short_filter_order: int = 3  # length of the explicit input convolutional filter
    outer_mixing: bool = False  # whether to use outer mixing
    activation: str = "gelu_new"  # activation function
    num_heads: int = 1  # Required for compatibility with other architectures, but not used directly
    mlp_scale: int = 4  # Scale factor for MLP dimensions

    # Filter parameters
    emb_dim: int = 3  # dim of input to MLP, augments with positional encoding
    use_fast_fft: bool = True  # whether to use fused FFT convolution
    filter_dropout: float = 0.0  # dropout probability for the filter
    fused_bias_fc: bool = False  # Whether to use fused bias in fully connected layers

    # Modulation parameters
    fast_decay_pct: float = 0.3
    slow_decay_pct: float = 1.5
    target: float = 1e-2
    modulate: bool = True
    shift: float = 0.0

    # General parameters
    initializer_range: float = 0.02
    embed_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5

    gradient_checkpointing: bool = True
    gradient_checkpointing_block_size: int = 5

    use_bias: bool = True
    post_order_ffn: bool = False
    return_state: bool = False  # Whether to return state information

    # Axes
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))

    @property
    def model_type(self) -> Type["HyenaLMHeadModel"]:
        return HyenaLMHeadModel



# JAX/Haliax implementations of the Hyena components

class ImplicitFilterMLP(eqx.Module):
    """MLP for the implicit filter in Hyena."""
    input_proj: hnn.Linear
    hidden_projs: list
    output_proj: hnn.Linear
    sin_activations: list

    @staticmethod
    def init(emb_dim: int, filter_order: int, embed_dim: int, use_bias: bool, num_hidden_layers: int = 2, *, key):
        keys = jrandom.split(key, num_hidden_layers + 2)

        # Define unique axes for each layer
        EmbDimAxis = Axis("emb_dim", emb_dim)
        FilterOrderAxis = Axis("filter_order", filter_order)

        # Input projection
        input_proj = hnn.Linear.init(
            In=EmbDimAxis,
            Out=FilterOrderAxis,
            key=keys[0],
            use_bias=use_bias
        )

        # Hidden projections
        hidden_projs = []
        sin_activations = []

        for i in range(num_hidden_layers):
            # Create a unique filter order axis for each hidden layer to avoid naming collisions
            InFilterAxis = Axis(f"filter_in_{i}", filter_order)
            OutFilterAxis = Axis(f"filter_out_{i}", filter_order)

            hidden_projs.append(
                hnn.Linear.init(
                    In=InFilterAxis,
                    Out=OutFilterAxis,
                    key=keys[i+1],
                    use_bias=use_bias
                )
            )
            sin_activations.append(Sin.init(filter_order, w=1))

        # Add an activation for the input projection
        sin_activations.insert(0, Sin.init(filter_order, w=1))

        # Output projection - use a different axis name to avoid embedding axis collision
        LastFilterAxis = Axis(f"filter_out_{num_hidden_layers-1}", filter_order)
        EmbedOutAxis = Axis("embed_out", embed_dim)
        output_proj = hnn.Linear.init(
            In=LastFilterAxis,
            Out=EmbedOutAxis,
            key=keys[-1],
            use_bias=False
        )

        return ImplicitFilterMLP(input_proj, hidden_projs, output_proj, sin_activations)

    def __call__(self, x, *, key=None):
        # Apply input projection
        x = self.input_proj(x)
        x = self.sin_activations[0](x)

        # Apply hidden layers
        for i, (proj, act) in enumerate(zip(self.hidden_projs, self.sin_activations[1:])):
            x = proj(x)
            x = act(x)

        # Apply output projection
        x = self.output_proj(x)

        return x


class PositionalEmbedding(eqx.Module):
    """Complex exponential positional embeddings for Hyena filters."""
    z: hax.NamedArray  # [1, seq_len, emb_dim]
    t: hax.NamedArray  # [1, seq_len, 1]

    @staticmethod
    def init(Pos: Axis, emb_dim: int, *, key=None):
        # The time embedding fed to the filters is normalized so that t_f = 1
        seq_len = Pos.size

        # Create plain JAX arrays for the positional embeddings
        # The shape is [1, seq_len, 1] for broadcasting
        t = jnp.zeros((1, seq_len, 1), dtype=jnp.float32)
        t = t + jnp.linspace(0, 1, seq_len).reshape(1, seq_len, 1)

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2

        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = jnp.zeros((1, seq_len, 1), dtype=jnp.float32)
        t_rescaled = t_rescaled + jnp.linspace(0, seq_len - 1, seq_len).reshape(1, seq_len, 1)

        w = 2 * jnp.pi * t_rescaled / seq_len

        f = jnp.linspace(1e-4, bands - 1, bands).reshape(1, 1, bands)
        z_complex = jnp.exp(-1j * f * w)
        z = jnp.concatenate([t, jnp.real(z_complex), jnp.imag(z_complex)], axis=-1)

        # Convert plain JAX arrays to NamedArrays
        Batch = Axis("batch", 1)
        EmDim = Axis("embedding_dim", z.shape[-1])
        z_named = hax.named(z, (Batch, Pos, EmDim))
        t_named = hax.named(t, (Batch, Pos, Axis("time_dim", 1)))

        return PositionalEmbedding(z_named, t_named)

    def __call__(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(eqx.Module):
    """Exponential modulation for the Hyena filter."""
    deltas: hax.NamedArray
    modulate: bool
    shift: float

    @staticmethod
    def init(Embed: Axis, fast_decay_pct: float, slow_decay_pct: float, target: float,
             modulate: bool, shift: float, *, key=None):
        max_decay = jnp.log(target) / fast_decay_pct
        min_decay = jnp.log(target) / slow_decay_pct
        deltas = jnp.linspace(min_decay, max_decay, Embed.size).reshape(1, 1, Embed.size)

        return ExponentialModulation(deltas, modulate, shift)

    def __call__(self, t, x):
        if self.modulate:
            decay = jnp.exp(-t * jnp.abs(self.deltas))
            x = x * (decay + self.shift)
        return x


class Sin(eqx.Module):
    """Sinusoidal activation function with trainable frequency."""
    freq: jnp.ndarray

    @staticmethod
    def init(dim: int, w: float = 10, train_freq: bool = True, *, key=None):
        freq = w * jnp.ones((1, dim))
        return Sin(freq)

    def __call__(self, x):
        return jnp.sin(self.freq * x)


def fft_conv(u, k, bias=None):
    """JAX implementation of FFT convolution."""
    seqlen = u.shape[-2]
    fft_size = 2 * seqlen

    # Reshape u and k for FFT
    u_f = jnp.fft.rfft(u, n=fft_size, axis=-2)
    k_f = jnp.fft.rfft(k, n=fft_size, axis=-2) / fft_size

    # Perform convolution in frequency domain
    y_f = u_f * k_f
    y = jnp.fft.irfft(y_f, n=fft_size, axis=-2)[..., :seqlen, :]

    # Add bias if provided
    if bias is not None:
        y = y + bias.reshape(1, 1, -1)

    return y


class HyenaFilter(eqx.Module):
    """Implicit long filter with modulation for Hyena."""
    implicit_filter: ImplicitFilterMLP
    modulation: ExponentialModulation
    pos_emb: PositionalEmbedding
    bias: jnp.ndarray
    normalized: bool
    use_bias: bool
    dropout: hnn.Dropout

    @staticmethod
    def init(config: HyenaConfig, *, key):
        keys = jrandom.split(key, 4)

        # Initialize positional embedding
        pos_emb = PositionalEmbedding.init(config.Pos, config.emb_dim, key=keys[0])

        # Initialize implicit filter MLP
        implicit_filter = ImplicitFilterMLP.init(
            emb_dim=config.emb_dim,
            filter_order=config.filter_order,
            embed_dim=config.hidden_dim,
            use_bias=config.use_bias,
            num_hidden_layers=2,
            key=keys[1]
        )

        # Initialize modulation
        modulation = ExponentialModulation.init(config.Embed,
                                               config.fast_decay_pct,
                                               config.slow_decay_pct,
                                               config.target,
                                               config.modulate,
                                               config.shift,
                                               key=keys[2])

        # Initialize bias
        bias = jrandom.normal(keys[3], (config.hidden_dim,))

        # Dropout
        dropout = hnn.Dropout(pdrop=config.filter_dropout)

        return HyenaFilter(implicit_filter, modulation, pos_emb, bias,
                           normalized=False, use_bias=config.use_bias, dropout=dropout)

    def filter(self, L, key=None):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)

        if self.normalized:
            h = h / jnp.norm(h, axis=-2, ord=1, keepdims=True)

        return h

    @named_call
    def __call__(self, x, L, k=None, bias=None, *, key=None):
        if k is None:
            k = self.filter(L, key=key)

        if bias is None:
            bias = self.bias

        bias = bias if self.use_bias else jnp.zeros_like(bias)

        # Reshape for FFT convolution
        batch_size, seq_len, channels = x.shape
        x_reshaped = x.reshape(batch_size, seq_len, channels)
        k_reshaped = k.reshape(1, seq_len, channels)

        # Apply FFT convolution
        y = fft_conv(x_reshaped, k_reshaped, bias)

        # Apply dropout
        if key is not None:
            y = self.dropout(y, key=key)

        return y


class HyenaOperator(eqx.Module):
    """Hyena operator - the core building block of the Hyena architecture."""
    config: HyenaConfig = eqx.static_field()
    in_proj: hnn.Linear
    out_proj: hnn.Linear
    short_filter: hnn.Conv
    filter_fn: HyenaFilter
    dropout: hnn.Dropout
    activation: Callable = eqx.static_field()

    @staticmethod
    def init(config: HyenaConfig, *, key):
        keys = jrandom.split(key, 5)

        # Input projection: d_model -> (order + 1) * d_model
        in_proj = hnn.Linear.init(
            In=config.Embed,
            Out=Axis("expanded", (config.order + 1) * config.hidden_dim),
            key=keys[0],
            use_bias=config.use_bias
        )

        # Output projection: d_model -> d_model
        # Create a new axis with the same dimensions to avoid naming collision
        OutputEmbed = Axis("output_embed", config.hidden_dim)
        out_proj = hnn.Linear.init(
            In=config.Embed,
            Out=OutputEmbed,
            key=keys[1],
            use_bias=config.use_bias
        )

        # Short filter (local convolution)
        total_width = config.hidden_dim * (config.order + 1)
        # Haliax Conv has a different signature than Conv1d, we need to adapt it
        InChannels = Axis("in_channels", total_width)
        OutChannels = Axis("out_channels", total_width)
        Spatial = Axis("position", config.seq_len)
        short_filter = hnn.Conv.init(
            Spatial,
            InChannels,
            OutChannels,
            kernel_size=config.short_filter_order,
            groups=total_width,
            padding=config.short_filter_order - 1,
            key=keys[2]
        )

        # Initialize the long-range filter
        filter_fn = HyenaFilter.init(config, key=keys[3])

        # Dropout
        dropout = hnn.Dropout(pdrop=config.resid_pdrop)

        # Activation function
        if config.activation == "gelu_new":
            activation = partial(hnn.gelu, approximate=True)
        elif config.activation == "gelu":
            activation = partial(hnn.gelu, approximate=False)
        elif config.activation == "relu":
            activation = hnn.relu
        elif config.activation == "silu" or config.activation == "swish":
            activation = hnn.silu
        else:
            activation = lambda x: x  # Identity

        return HyenaOperator(
            config=config,
            in_proj=in_proj,
            out_proj=out_proj,
            short_filter=short_filter,
            filter_fn=filter_fn,
            dropout=dropout,
            activation=activation
        )

    @named_call
    def __call__(self, u, *, key=None):
        k1, k2 = haliax.jax_utils.maybe_rng_split(key, 2) if key is not None else (None, None)

        # Input projection
        u = self.in_proj(u, key=k1)

        # Reshape for processing
        batch_size, seq_len, total_channels = u.shape
        u = u.reshape(batch_size, seq_len, self.config.order + 1, self.config.hidden_dim)

        # Apply short filter (local convolution)
        u_reshaped = u.reshape(batch_size, seq_len, -1)
        u_for_conv = u_reshaped.transpose(0, 2, 1)  # [batch, channels, seq_len]

        # Apply convolution
        u_short_conv = self.short_filter(u_for_conv)

        # Transpose back to [batch, seq_len, channels]
        u_short = u_short_conv.transpose(0, 2, 1)
        u_short = u_short.reshape(batch_size, seq_len, self.config.order + 1, self.config.hidden_dim)

        # Extract the components
        *x, v = jnp.split(u_short, self.config.order + 1, axis=2)
        v = v.squeeze(2)  # Remove the singleton dimension

        # Long-range filtering with recurrence
        for o, x_i in enumerate(reversed(x[1:])):
            # Compress the order dimension
            x_i = x_i.squeeze(2)

            # Apply mixing
            if self.config.outer_mixing:
                v = jnp.einsum('bsi,bsj->bsij', v, x_i)
                v = self.dropout(v, key=k2)
                v = v.sum(axis=-1)  # Sum over the inner dimension
            else:
                v = self.dropout(v * x_i, key=k2)

            # Apply filtering
            v = self.filter_fn(v, seq_len, key=k2)

            # Apply additional MLP if configured
            if self.config.post_order_ffn:
                # This would require additional implementation for the case when post_order_ffn=True
                pass

        # First x is special - element-wise product with the filtered result
        y = self.activation(v * x[0].squeeze(2))

        # Output projection
        y = self.out_proj(y, key=k2)

        return y


# Create a proper MLP class similar to Gpt2Mlp
class HyenaMlp(ModuleWithStateDictSerialization, eqx.Module):
    c_fc: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    c_proj: hnn.Linear  # projection from Intermediate to Embed
    act: Callable = eqx.static_field()

    @staticmethod
    def init(Embed: Axis, Mlp: Axis, activation_fn, *, key, use_bias: bool = True) -> "HyenaMlp":
        k_fc, k_proj = jrandom.split(key, 2)
        c_fc = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=False)
        c_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_proj, use_bias=use_bias, out_first=False)

        # Handle activation function
        if isinstance(activation_fn, str):
            if activation_fn == "gelu_new":
                act = partial(hnn.gelu, approximate=True)
            elif activation_fn == "gelu":
                act = partial(hnn.gelu, approximate=False)
            elif activation_fn == "relu":
                act = hnn.relu
            elif activation_fn == "silu" or activation_fn == "swish":
                act = hnn.silu
            else:
                act = lambda x: x  # Identity
        else:
            act = activation_fn

        return HyenaMlp(c_fc, c_proj, act)

    @named_call
    def __call__(self, x: NamedArray, *, key=None):
        k1, k2 = hax.jax_utils.maybe_rng_split(key, 2)
        x = self.c_fc(x, key=k1)
        x = self.act(x)
        x = self.c_proj(x, key=k2)
        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"c_fc": "c_fc", "c_proj": "c_proj"}


class HyenaBlock(eqx.Module):
    """A Hyena block similar to a transformer block but with the HyenaOperator instead of attention."""
    ln_1: hnn.LayerNorm
    hyena: HyenaOperator
    ln_2: hnn.LayerNorm
    mlp: HyenaMlp
    resid_dropout: hnn.Dropout

    @staticmethod
    def init(config: HyenaConfig, *, key):
        keys = jrandom.split(key, 3)

        ln_1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        hyena = HyenaOperator.init(config, key=keys[0])
        ln_2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        # MLP similar to GPT2
        Mlp = Axis("mlp", size=config.hidden_dim * config.mlp_scale)  # Using the configured mlp_scale

        # Initialize the MLP properly using our new class
        mlp = HyenaMlp.init(
            config.Embed,
            Mlp,
            config.activation,
            key=keys[1],
            use_bias=config.use_bias
        )

        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)

        return HyenaBlock(ln_1, hyena, ln_2, mlp, resid_dropout)

    @named_call
    def __call__(self, x, *, key=None):
        k1, k2, k3, k4 = hax.jax_utils.maybe_rng_split(key, 4)

        # First sub-block: Hyena
        hyena_output = self.hyena(self.ln_1(x), key=k1)
        hyena_output = self.resid_dropout(hyena_output, key=k2)
        x = x + hyena_output

        # Second sub-block: MLP
        ff_output = self.mlp(self.ln_2(x), key=k3)
        ff_output = self.resid_dropout(ff_output, key=k4)
        x = x + ff_output

        return x


class HyenaEmbeddings(ModuleWithStateDictSerialization, eqx.Module):
    """Token and position embeddings for Hyena, similar to GPT2."""
    Vocab: Axis = eqx.static_field()
    config: HyenaConfig = eqx.static_field()

    token_embeddings: hnn.Embedding
    position_embeddings: hnn.Embedding
    dropout: hnn.Dropout

    @staticmethod
    def init(Vocab: Axis, config: HyenaConfig, *, key):
        k_wte, k_wpe = jrandom.split(key, 2)

        token_embeddings = hnn.Embedding.init(
            Vocab, config.Embed, key=k_wte, initializer_range=config.initializer_range
        )
        position_embeddings = hnn.Embedding.init(
            config.Pos, config.Embed, key=k_wpe, initializer_range=config.initializer_range/2
        )
        dropout = hnn.Dropout(pdrop=config.embed_pdrop)

        return HyenaEmbeddings(Vocab, config, token_embeddings, position_embeddings, dropout)

    @named_call
    def embed(self, input_ids, *, key):
        input_embeds = self.token_embeddings(input_ids)
        input_Pos = input_ids.resolve_axis("position")
        position_embeds = self.position_embeddings.embed(hax.arange(input_Pos))
        x = input_embeds + position_embeds
        x = self.dropout(x, key=key)

        return x

    def unembed(self, x: NamedArray):
        return hax.dot(x, self.token_embeddings.weight, axis="embed")

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "wte", "position_embeddings": "wpe"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_token_embeddings = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_token_embeddings)


class HyenaTransformer(ModuleWithStateDictSerialization):
    """The main transformer model with Hyena blocks instead of attention blocks."""
    config: HyenaConfig = eqx.static_field()
    blocks: Stacked[HyenaBlock]
    ln_f: hnn.LayerNorm

    @staticmethod
    def init(config: HyenaConfig, *, key):
        # Vectorize the blocks using Stacked
        blocks = Stacked.init(config.Layers, HyenaBlock, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return HyenaTransformer(config, blocks, ln_f)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None

        # Process through all blocks
        # Unlike GPT2, we don't need to pass attention masks to Hyena blocks
        x = self.blocks.fold(x, key=keys)
        x = self.ln_f(x)

        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"blocks": "h"}


class HyenaLMHeadModel(LmWithHfSerializationMixin[HyenaConfig]):
    """Hyena Language Model with a head for predicting next tokens."""
    transformer: HyenaTransformer
    embeddings: HyenaEmbeddings

    @property
    def config(self):
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @classmethod
    def init(cls, Vocab: Axis, config: HyenaConfig, *, key) -> "HyenaLMHeadModel":
        k_t, k_embeddings = jrandom.split(key, 2)
        transformer = HyenaTransformer.init(config, key=k_t)
        embeddings = HyenaEmbeddings.init(Vocab, config, key=k_embeddings)

        return HyenaLMHeadModel(transformer, embeddings)

    def activations(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        k_embed, k_transformer = haliax.jax_utils.maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids, key=k_embed)
        # Hyena doesn't need attention masks, so we don't pass them
        x = self.transformer(x, key=k_transformer)

        return x

    def get_lm_head(self) -> hax.NamedArray:
        return self.embeddings.token_embeddings.weight

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "HyenaLMHeadModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": None, "embeddings": None}

