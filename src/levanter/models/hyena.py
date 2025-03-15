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
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray
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
    EmbedOrderMinus1 = property(lambda self: Axis(name="embed_order_minus_1", size=self.hidden_dim * (self.order - 1)))
    EmbedOrderPlus1 = property(lambda self: Axis(name="embed_order_plus_1", size=self.hidden_dim * (self.order + 1)))
    OrderMinus1 = property(lambda self: Axis(name="order_minus_1", size=self.order - 1))

    def __post_init__(self):
        if self.seq_len % self.num_blocks:
            raise ValueError(f"seq_len {self.seq_len} must be divisible by num_blocks {self.num_blocks}")


class PositionalEmbedding(eqx.Module):
    """Complex exponential positional embeddings for Hyena filters."""

    z: hax.NamedArray  # [Pos, Embed]
    t: hax.NamedArray  # [Pos]
    Pos: Axis = eqx.field(static=True)

    @staticmethod
    def init(Pos: Axis, Embed: Axis, *, key=None):
        """Initialize positional embeddings for Hyena filters.

        Args:
            Pos: Position axis, will be in the outputs of __call__.
            Embed: Axis of positional embedding
            key: Optional random key (not used)
        """
        seq_len = Pos.size

        # Ensure emb_dim is valid
        if Embed.size <= 1:
            raise ValueError("emb_dim must be greater than 1")

        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = hax.linspace(Pos, start=0, stop=1)

        # Calculate number of frequency bands
        bands = (Embed.size - 1) // 2
        Band = Axis("band", bands)

        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = hax.linspace(Pos, start=0, stop=seq_len - 1)
        w = 2 * jnp.pi * t_rescaled / seq_len

        f = hax.linspace(Band, start=1e-4, stop=bands - 1)
        z_complex = hax.exp(-1j * f.broadcast_axis(Pos) * w.broadcast_axis(Band))

        # Concatenate time and complex components
        z = hax.concatenate(Band, [t.broadcast_axis(Band), hax.real(z_complex), hax.imag(z_complex)])
        assert z.resolve_axis(Band).size == Embed.size
        z = hax.rename(z, {Band: Embed})

        return PositionalEmbedding(z, t, Pos)

    def __call__(self, L):
        """Get positional embeddings for the first L positions.

        Args:
            L: Length to get embeddings for

        Returns:
            Tuple of (z, t) embeddings limited to length L
        """
        if L > self.Pos.size:
            raise ValueError(f"Requested length {L} > max size {self.Pos.size}")

        return self.z.slice(self.Pos, length=L), self.t.slice(self.Pos, length=L)


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


def fft_conv(u: jax.Array, k: jax.Array, bias=None) -> jax.Array:
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

        pos_emb = PositionalEmbedding.init(config.Pos, config.FilterEmbed, key=keys[0])

        implicit_filter = hax.nn.MLP.init(
            Input=config.FilterEmbed,
            width=config.FilterOrder,
            Output=config.EmbedOrderMinus1,
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

    def generate_filters(self, input_length: int, *, key=None) -> hax.NamedArray:
        """Generate filter kernels for Hyena operation.

        Args:
            input_length: Length of input sequence
            key: Optional PRNG key for dropout

        Returns:
            NamedArray containing filter
        """
        z, t = self.pos_emb(input_length)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)

        if self.normalized:
            # Implement L1 norm manually since there's no haliax.norm
            h_abs = hax.abs(h)
            norm_values = hax.sum(h_abs, axis=h.axes[1], where=None)
            h = h / norm_values.broadcast_axis(h.axes[1])

        return h

    @named_call
    def __call__(self, x: hax.NamedArray, k: hax.NamedArray, bias=None, *, key=None):
        """Apply the hyena filter.

        Args:
            x: Input tensor with shape (batch, seq_len, channels)
            k: Optional filter to use
            bias: Optional bias to use (if None, uses self.bias)
            key: Optional PRNG key for dropout

        Returns:
            Filtered tensor with same shape as input
        """
        if bias is None:
            bias = self.bias

        bias = bias if self.use_bias else jnp.zeros_like(bias)

        # fft_conv is not haliax aware so we have to rearrange and pass in raw arrays.
        x_arr = hax.rearrange(x, ()).array  # TODO
        k_arr = hax.rearrange(k, ()).array  # TODO

        y_arr = fft_conv(x_arr, k_arr, bias)
        y = hax.named(y_arr, x.axes)

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
            Out=config.EmbedOrderPlus1,
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
            In=config.EmbedOrderPlus1,
            Out=config.EmbedOrderPlus1.alias("out_channels"),
            kernel_size=config.short_filter_order,
            groups=config.EmbedOrderPlus1.size,
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
    def __call__(self, u: hax.NamedArray, *, key: PRNGKeyArray | None = None) -> hax.NamedArray:
        key_in_proj, key_dropout = haliax.jax_utils.maybe_rng_split(key, 2)
        Pos = self.config.Pos
        Block = self.config.Block
        PosPerBlock = self.config.PosPerBlock
        EmbedOrderPlus1 = self.config.EmbedOrderPlus1
        Embed = self.config.Embed
        # input has the same axis name as the Pos axis, but possibly different size
        input_length = u.axis_size(Pos.name)
        l_filter = min(input_length, Pos.size)

        # Input projection from [Embed] to [(order+1) * Embed]
        u = self.in_proj(u, key=key_in_proj)

        # trying to keep the variable names from the official impl.
        # I think uc stands for "u convolved".
        uc = self.short_filter(u).rename({"out_channels": EmbedOrderPlus1})
        uc = uc.slice(Pos, length=l_filter)

        # Now we need to reshape to match the PyTorch implementation's:
        # 'b (ho v) (z l) -> b ho v z l'
        # we don't have b (we vmap over it).
        # we don't have ho (hard-coding num_heads to 1)
        uc = hax.unflatten_axis(uc, Pos, (Block, PosPerBlock))

        components = hax.split(uc, EmbedOrderPlus1, [Embed] * (self.config.order + 1))
        v = components[-1]
        x = components[:-1]
        assert len(x) == self.config.order
        filters = self.filter_fn.generate_filters(l_filter, key=key_dropout)
        filters = hax.unflatten_axis(
            filters, self.config.EmbedOrderMinus1, (self.config.OrderMinus1, self.config.Embed)
        )
        filters_list = filters.unbind(self.config.OrderMinus1)

        # Long-range filtering with recurrence
        for filter_order, x_i in enumerate(reversed(x[1:])):
            # Outer product of Embed with Embed
            if self.config.outer_mixing:
                EmbedPrime = self.config.Embed.alias("embed_prime")
                v_for_outer = hax.rename(v, {Embed: EmbedPrime})
                outer_product = v_for_outer.broadcast_axis(EmbedPrime) * x_i
                v = self.dropout(outer_product, key=key_dropout)
                v = hax.sum(v, EmbedPrime)
            else:
                v = self.dropout(v * x_i, key=key_dropout)

            v = self.filter_fn(v, filters_list[filter_order], key=key_dropout)

            # Not currently supporting the post_order_ffn from the PyTorch impl.

        v = v * x[0]
        # flatten the block axis
        v = hax.rearrange(
            v,
            (
                f"{EmbedOrderPlus1.name} {Block.name} {PosPerBlock.name} ->"
                f"{EmbedOrderPlus1.name} ({Pos.name}: {Block.name} {PosPerBlock.name})"
            ),
        )
        y = self.activation(v)

        # Output projection
        y = self.out_proj(y, key=key_dropout)

        return y
