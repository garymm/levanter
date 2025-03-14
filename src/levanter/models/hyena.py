"""An implementation of the Hyena operator.

Paper: [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)
Official implementation in PyTorch:
https://github.com/HazyResearch/safari/blob/541902aca88cb11af4d816ac762f3303e4ff8eea/src/models/sequence/hyena.py

Current diffences from the official impl:
- We don't support inner_factor.
- We don't support post_order_ffn.
- We don't support num_heads (the PyTorch impl's support for multiple heads seems incomplete, or at least
  I don't understand it).
"""

from dataclasses import dataclass
import typing

import equinox as eqx
import haliax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis
from haliax.jax_utils import named_call

from levanter.utils.activation import ActivationFunction, ActivationFunctionName


@dataclass(frozen=True)
class HyenaConfig:
    seq_len: int = 1024  # l_max from PyTorch impl
    hidden_dim: int = 768  # d_model from PyTorch impl

    # hyena specific parameters
    order: int = 2  # depth of the Hyena recurrence
    filter_order: int = 16  # width of the FFN parametrizing the implicit filter
    short_filter_order: int = 3  # length of the explicit input convolutional filter
    outer_mixing: bool = False  # whether to use outer mixing
    activation: ActivationFunctionName = ActivationFunctionName.GELU_NEW
    num_blocks: int = 1  # number of blocks to split the sequence into
    num_hidden_layers_filter_mlp: int = 2  # number of inner linear layers inside filter MLP

    # Filter parameters
    filter_emb_dim: int = 3  # dim of input to MLP, augments with positional encoding
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

    # Axes
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Block = property(lambda self: Axis(name="blocks", size=self.num_blocks))
    PosPerBlock = property(lambda self: Axis(name="pos_per_block", size=self.seq_len // self.num_blocks))
    FilterOrder = property(lambda self: Axis(name="filter_order", size=self.filter_order))
    FilterEmbed = property(lambda self: Axis(name="filter_embed", size=self.filter_emb_dim))
    EmbedAllOrders = property(lambda self: Axis(name="embed_all_orders", size=(self.order + 1) * self.hidden_dim))

    def __post_init__(self):
        if self.seq_len % self.num_blocks:
            raise ValueError(f"seq_len {self.seq_len} must be divisible by num_blocks {self.num_blocks}")


class PositionalEmbedding(eqx.Module):
    """Complex exponential positional embeddings for Hyena filters."""

    z: hax.NamedArray  # [seq_len, emb_dim]
    t: hax.NamedArray  # [seq_len, 1]
    Pos: Axis = eqx.field(static=True)
    EmDim: Axis = eqx.field(static=True)
    TimeDim: Axis = eqx.field(static=True)

    @staticmethod
    def init(Pos: Axis, emb_dim: int, *, key=None):
        """Initialize positional embeddings for Hyena filters.

        Args:
            Pos: Position axis with size equal to seq_len
            emb_dim: Dimension of positional embedding
            key: Optional random key (not used)
        """
        seq_len = Pos.size

        # Ensure emb_dim is valid
        if emb_dim <= 1:
            raise ValueError("emb_dim must be greater than 1")

        # Calculate number of frequency bands
        bands = (emb_dim - 1) // 2

        # Create time embedding normalized to [0, 1]
        t_array = jnp.linspace(0, 1, seq_len).reshape(seq_len, 1)

        # Create rescaled time for frequencies
        t_rescaled = jnp.linspace(0, seq_len - 1, seq_len).reshape(seq_len, 1)

        # Calculate frequencies
        w = 2 * jnp.pi * t_rescaled / seq_len

        # Create frequency bands
        f = jnp.linspace(1e-4, bands - 1, bands).reshape(1, bands)
        z_complex = jnp.exp(-1j * f * w)

        # Concatenate time and complex components
        z_array = jnp.concatenate([t_array, jnp.real(z_complex), jnp.imag(z_complex)], axis=-1)

        # Create Haliax axes
        EmDim = Axis("embedding_dim", z_array.shape[-1])
        TimeDim = Axis("time_dim", 1)

        # Create named arrays
        z_named = hax.named(z_array, (Pos, EmDim))
        t_named = hax.named(t_array, (Pos, TimeDim))

        return PositionalEmbedding(z_named, t_named, Pos, EmDim, TimeDim)

    def __call__(self, L):
        """Get positional embeddings for the first L positions.

        Args:
            L: Length to get embeddings for

        Returns:
            Tuple of (z, t) embeddings limited to length L
        """
        if L > self.Pos.size:
            raise ValueError(f"Requested length {L} exceeds maximum length {self.Pos.size}")

        # Create a subset axis of length L
        L_pos = Axis("position", L)

        indices = hax.arange(L_pos)

        # Map from L_pos to Pos for indexing
        z_subset = self.z[{self.Pos.name: indices}]
        t_subset = self.t[{self.Pos.name: indices}]

        return z_subset, t_subset


class ExponentialModulation(eqx.Module):
    """Exponential modulation for the Hyena filter."""

    deltas: hax.NamedArray
    modulate: bool
    shift: float
    Embed: Axis = eqx.field(static=True)

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
        """Initialize exponential modulation for Hyena filter.

        Args:
            Embed: Embedding dimension axis
            fast_decay_pct: Fast decay percentage
            slow_decay_pct: Slow decay percentage
            target: Target value for decay
            modulate: Whether to apply modulation
            shift: Shift value for modulation
            key: Random key (not used)
        """
        max_decay = jnp.log(target) / fast_decay_pct
        min_decay = jnp.log(target) / slow_decay_pct

        # Create deltas array directly as a named array
        decays = jnp.linspace(min_decay, max_decay, Embed.size)
        deltas = hax.named(decays, (Embed,))

        return ExponentialModulation(deltas, modulate, shift, Embed)

    def __call__(self, t, x):
        """Apply exponential modulation to input.

        Args:
            t: Time values
            x: Input tensor to modulate

        Returns:
            Modulated tensor
        """
        if self.modulate:
            # Apply modulation using Haliax operations
            deltas_abs = hax.abs(self.deltas)

            # t should be a NamedArray with a time dimension
            # We need to broadcast it against deltas_abs
            decay = hax.exp(-t * deltas_abs.broadcast_to(x.axes))

            x = x * (decay + self.shift)

        return x


class Sin(eqx.Module):
    """Sinusoidal activation function with trainable frequency."""

    freq: hax.NamedArray

    @staticmethod
    def init(Order: Axis, w: float = 10, *, key=None):
        return Sin(w * hax.ones((Order,)))

    def __call__(self, x: hax.NamedArray) -> hax.NamedArray:
        return hax.sin(self.freq * x)


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

    implicit_filter: hax.nn.MLP
    modulation: ExponentialModulation
    pos_emb: PositionalEmbedding
    bias: jnp.ndarray
    normalized: bool
    use_bias: bool
    dropout: hnn.Dropout

    @staticmethod
    def init(config: HyenaConfig, *, key):
        keys = jrandom.split(key, 4)

        pos_emb = PositionalEmbedding.init(config.Pos, config.filter_emb_dim, key=keys[0])

        implicit_filter = hax.nn.MLP.init(
            Input=config.FilterEmbed,
            width=config.FilterOrder,
            Output=config.Embed,
            depth=config.num_hidden_layers_filter_mlp,
            activation=Sin.init(config.FilterOrder, w=1),
            use_bias=config.use_bias,
            key=keys[1],
        )

        modulation = ExponentialModulation.init(
            config.Embed,
            config.fast_decay_pct,
            config.slow_decay_pct,
            config.target,
            config.modulate,
            config.shift,
            key=keys[2],
        )

        bias = jrandom.normal(keys[3], (config.hidden_dim,))

        dropout = hnn.Dropout(pdrop=config.filter_dropout)

        return HyenaFilter(
            implicit_filter, modulation, pos_emb, bias, normalized=False, use_bias=config.use_bias, dropout=dropout
        )

    def filter(self, L, key=None):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)

        if self.normalized:
            # Implement L1 norm manually since there's no haliax.norm
            h_abs = hax.abs(h)
            norm_values = hax.sum(h_abs, axis=h.axes[1], where=None)
            h = h / norm_values.broadcast_axis(h.axes[1])

        return h

    @named_call
    def __call__(self, x, L, k=None, bias=None, *, key=None):
        """Apply the hyena filter.

        Args:
            x: Input tensor with shape (batch, seq_len, channels)
            L: Sequence length
            k: Optional filter to use (if None, computed using self.filter)
            bias: Optional bias to use (if None, uses self.bias)
            key: Optional PRNG key for dropout

        Returns:
            Filtered tensor with same shape as input
        """
        if k is None:
            k = self.filter(L, key=key)

        if bias is None:
            bias = self.bias

        bias = bias if self.use_bias else jnp.zeros_like(bias)

        # Apply FFT convolution with support for NamedArrays
        y = fft_conv(x, k, bias)

        # Apply dropout if key is provided
        if key is not None and self.dropout.pdrop > 0:
            dropout_key = haliax.jax_utils.maybe_rng_split(key, 1)[0]
            y = self.dropout(y, key=dropout_key)

        return y


class HyenaOperator(eqx.Module):
    """Hyena operator - the core building block of the Hyena architecture."""

    config: HyenaConfig = eqx.field(static=True)
    in_proj: hnn.Linear
    out_proj: hnn.Linear
    short_filter: hnn.Conv
    filter_fn: HyenaFilter
    dropout: hnn.Dropout
    activation: ActivationFunction = eqx.field(static=True)

    @staticmethod
    def init(config: HyenaConfig, *, key):
        keys = jrandom.split(key, 5)

        in_proj = hnn.Linear.init(
            In=config.Embed,
            Out=config.EmbedAllOrders,
            key=keys[0],
            use_bias=config.use_bias,
        )

        # Output projection: hidden_size -> hidden_size
        # Create a new axis with the same dimensions to avoid naming collision
        # We do not support inner_factor from the PyTorch impl.
        out_proj = hnn.Linear.init(
            In=config.Embed, Out=config.Embed.alias("output_embed"), key=keys[1], use_bias=config.use_bias
        )

        short_filter = hnn.Conv.init(
            Spatial=config.Pos,
            In=config.EmbedAllOrders,
            Out=config.EmbedAllOrders.alias("out_channels"),
            kernel_size=config.short_filter_order,
            groups=config.EmbedAllOrders.size,
            padding=config.short_filter_order - 1,
            key=keys[2],
        )

        filter_fn = HyenaFilter.init(config, key=keys[3])
        dropout = hnn.Dropout(pdrop=config.resid_pdrop)

        return HyenaOperator(
            config=config,
            in_proj=in_proj,
            out_proj=out_proj,
            short_filter=short_filter,
            filter_fn=filter_fn,
            dropout=dropout,
            activation=config.activation.to_fn(),
        )

    @named_call
    def __call__(self, u, *, key=None):
        key_in_proj, key_dropout = haliax.jax_utils.maybe_rng_split(key, 2)

        assert u.resolve_axis("position") == self.config.Pos
        Pos = self.config.Pos
        Block = self.config.Block
        PosPerBlock = self.config.PosPerBlock
        EmbedAllOrders = self.config.EmbedAllOrders

        # Input projection from [Embed] to [(order+1) * Embed]
        u = self.in_proj(u, key=key_in_proj)

        # trying to keep the variable names from the official impl.
        # I think uc stands for "u convolved".
        uc = self.short_filter(u).rename({"out_channels": EmbedAllOrders})

        # Now we need to reshape to match the PyTorch implementation's:
        # 'b (ho v) (z l) -> b ho v z l'
        # we don't have b (we vmap over it).
        # we don't have ho (hard-coding num_heads to 1)

        uc = hax.unflatten_axis(uc, Pos, (Block, PosPerBlock))

        # Extract the components - we'll unbind on the Order axis
        components = hax.unbind(uc, EmbedAllOrders)
        v = components[-1]  # Last component is v
        x = components[:-1]  # All others are x components
        assert len(x) == self.config.order

        # Long-range filtering with recurrence
        for x_i in reversed(x[1:]):
            # Outer product of EmbedAllOrders with EmbedAllOrders
            if self.config.outer_mixing:
                EmbedAllOrdersPrime = EmbedAllOrders.alias("embed_all_orders_prime")
                v_for_outer = hax.rename(typing.cast(hax.NamedArray, v), {EmbedAllOrders: EmbedAllOrdersPrime})
                outer_product = v_for_outer.broadcast_axis(EmbedAllOrdersPrime) * x_i
                v = self.dropout(outer_product, key=key_dropout)
                v = hax.sum(v, EmbedAllOrdersPrime)
            else:
                v = self.dropout(v * x_i, key=key_dropout)

            # Apply filtering
            seq_len = Pos.size
            v = self.filter_fn(v, seq_len, key=key_dropout)

            # Not currently supporting the post_order_ffn from the PyTorch impl.

        v = v * x[0]
        # flatten the block axis
        v = hax.rearrange(
            typing.cast(hax.NamedArray, v),
            (
                f"{EmbedAllOrders.name} {Block.name} {PosPerBlock.name} ->"
                f"{EmbedAllOrders.name} ({Pos.name}: {Block.name} {PosPerBlock.name})"
            ),
        )
        y = self.activation(v)

        # Output projection
        y = self.out_proj(y, key=key_dropout)

        return y
