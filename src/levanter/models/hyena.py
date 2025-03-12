"""An implementation of the Hyena operator.

Paper: [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)
Official implementation in PyTorch:
https://github.com/HazyResearch/safari/blob/541902aca88cb11af4d816ac762f3303e4ff8eea/src/models/sequence/hyena.py
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import equinox as eqx
import haliax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis
from haliax.jax_utils import named_call


@dataclass(frozen=True)
class HyenaConfig:
    seq_len: int = 1024
    hidden_dim: int = 768

    # hyena specific parameters
    order: int = 2  # depth of the Hyena recurrence
    filter_order: int = 64  # width of the FFN parametrizing the implicit filter
    inner_factor: int = 1  # inner dimension multiplier
    short_filter_order: int = 3  # length of the explicit input convolutional filter
    outer_mixing: bool = False  # whether to use outer mixing
    activation: str = "gelu_new"  # activation function

    # Filter parameters
    emb_dim: int = 3  # dim of input to MLP, augments with positional encoding
    filter_dropout: float = 0.0  # dropout probability for the filter

    # Modulation parameters
    fast_decay_pct: float = 0.3
    slow_decay_pct: float = 1.5
    target: float = 1e-2
    modulate: bool = True
    shift: float = 0.0

    # General parameters
    resid_pdrop: float = 0.0  # Dropout for residual connections
    use_bias: bool = True  # Whether to use bias in linear layers
    post_order_ffn: bool = False  # Apply a dense layer between steps of the recurrence
    return_state: bool = False  # Whether to return state information

    # Axes
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))


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
        input_proj = hnn.Linear.init(In=EmbDimAxis, Out=FilterOrderAxis, key=keys[0], use_bias=use_bias)

        # Hidden projections
        hidden_projs = []
        sin_activations = []

        for i in range(num_hidden_layers):
            # Create a unique filter order axis for each hidden layer to avoid naming collisions
            InFilterAxis = Axis(f"filter_in_{i}", filter_order)
            OutFilterAxis = Axis(f"filter_out_{i}", filter_order)

            hidden_projs.append(
                hnn.Linear.init(In=InFilterAxis, Out=OutFilterAxis, key=keys[i + 1], use_bias=use_bias)
            )
            sin_activations.append(Sin.init(filter_order, w=1))

        # Add an activation for the input projection
        sin_activations.insert(0, Sin.init(filter_order, w=1))

        # Output projection - use a different axis name to avoid embedding axis collision
        LastFilterAxis = Axis(f"filter_out_{num_hidden_layers-1}", filter_order)
        EmbedOutAxis = Axis("embed_out", embed_dim)
        output_proj = hnn.Linear.init(In=LastFilterAxis, Out=EmbedOutAxis, key=keys[-1], use_bias=False)

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
    def init(
        Embed: Axis,
        fast_decay_pct: float,
        slow_decay_pct: float,
        target: float,
        modulate: bool,
        shift: float,
        *,
        key=None,
    ):
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
            key=keys[1],
        )

        # Initialize modulation
        modulation = ExponentialModulation.init(
            config.Embed,
            config.fast_decay_pct,
            config.slow_decay_pct,
            config.target,
            config.modulate,
            config.shift,
            key=keys[2],
        )

        # Initialize bias
        bias = jrandom.normal(keys[3], (config.hidden_dim,))

        # Dropout
        dropout = hnn.Dropout(pdrop=config.filter_dropout)

        return HyenaFilter(
            implicit_filter, modulation, pos_emb, bias, normalized=False, use_bias=config.use_bias, dropout=dropout
        )

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
            use_bias=config.use_bias,
        )

        # Output projection: d_model -> d_model
        # Create a new axis with the same dimensions to avoid naming collision
        OutputEmbed = Axis("output_embed", config.hidden_dim)
        out_proj = hnn.Linear.init(In=config.Embed, Out=OutputEmbed, key=keys[1], use_bias=config.use_bias)

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
            key=keys[2],
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
            activation=activation,
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
                v = jnp.einsum("bsi,bsj->bsij", v, x_i)
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
