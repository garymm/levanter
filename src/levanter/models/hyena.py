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

from levanter.utils.activation import ActivationFunction, ActivationFunctionName


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
    activation: ActivationFunctionName = ActivationFunctionName.GELU_NEW
    num_blocks: int = 1  # number of blocks to split the sequence into
    num_heads: int = 1  # number of heads

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
    return_state: bool = False  # Whether to return state information

    # Axes
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Block = property(lambda self: Axis(name="blocks", size=self.num_blocks))
    PosPerBlock = property(lambda self: Axis(name="pos_per_block", size=self.seq_len // self.num_blocks))
    Head = property(lambda self: Axis(name="head", size=self.num_heads))
    EmbedPerHead = property(lambda self: Axis(name="embed_per_head", size=self.hidden_dim // self.num_heads))

    def __post_init__(self):
        if self.seq_len % self.num_blocks:
            raise ValueError(f"seq_len {self.seq_len} must be divisible by num_blocks {self.num_blocks}")
        if self.hidden_dim % self.num_heads:
            raise ValueError(f"hidden_dim {self.hidden_dim} must be divisible by num_heads {self.num_heads}")


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
            # Implement L1 norm manually since jnp.norm isn't recognized
            # Compute sum of absolute values along the sequence dimension
            h_abs = hax.abs(h)
            # Use where parameter instead of keepdims
            norm_values = hax.sum(h_abs, axis=h.axes[1], where=None)
            # Reshape and broadcast the norm values
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

        # Input projection: d_model -> (order + 1) * d_model
        in_proj = hnn.Linear.init(
            In=config.Embed,
            Out=Axis("embed_all_orders", (config.order + 1) * config.hidden_dim),
            key=keys[0],
            use_bias=config.use_bias,
        )

        # Output projection: d_model -> d_model
        # Create a new axis with the same dimensions to avoid naming collision
        OutputEmbed = Axis("output_embed", config.hidden_dim)
        out_proj = hnn.Linear.init(In=config.Embed, Out=OutputEmbed, key=keys[1], use_bias=config.use_bias)

        # Short filter (local convolution)
        total_width = config.hidden_dim * (config.order + 1)
        short_filter = hnn.Conv.init(
            Spatial=config.Pos,
            In=Axis("in_channels", total_width),
            Out=Axis("out_channels", total_width),
            kernel_size=config.short_filter_order,
            groups=total_width,
            padding=config.short_filter_order - 1,
            key=keys[2],
        )

        # Initialize the long-range filter
        filter_fn = HyenaFilter.init(config, key=keys[3])

        # Dropout
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
        Order = Axis("order", self.config.order + 1)
        Embed = self.config.Embed

        # Create axes for blocks and other dimensions
        Block = self.config.Block
        PosPerBlock = Axis("pos_per_block", self.config.seq_len // self.config.num_blocks)
        Head = self.config.Head
        EmbedPerHead = self.config.EmbedPerHead

        # Input projection from [Embed] to [(order+1) * Embed]
        u = self.in_proj(u, key=key_in_proj)

        # For the short filter, we need to reshape to a format where channels come first
        EmbedAllOrders = self.in_proj.Out
        u = u.rearrange((EmbedAllOrders, Pos))
        # trying to keep the variable names from the official impl.
        # I think uc stands for "u convolved".
        uc = self.short_filter(u)

        # Now we need to reshape to match the PyTorch implementation's:
        # 'b (ho v) (z l) -> b ho v z l'
        # we don't have b (we vmap over it).

        uc = hax.unflatten_axis(uc, Embed, (Head, EmbedPerHead))
        uc = hax.unflatten_axis(uc, Pos, (Block, PosPerBlock))

        # Extract the components - we'll unbind on the Order axis
        components = hax.unbind(uc, EmbedPerHead)
        v = components[-1]  # Last component is v
        x = components[:-1]  # All others are x components
        assert len(x) == self.config.order

        # Long-range filtering with recurrence
        # TODO: not done. Go through torch impl and compare.
        for o, x_i in enumerate(reversed(x[1:])):
            # Apply mixing, handling outer_mixing with Haliax operations
            if self.config.outer_mixing:
                # Get the dimension sizes directly from the NamedArrays
                v_embed_axis = v.axes[-1]
                x_embed_axis = x_i.axes[-1]

                # Create axes for the outer product
                V_dim = Axis("v_dim", v_embed_axis.size)
                X_dim = Axis("x_dim", x_embed_axis.size)

                # Broadcast to prepare for outer product
                v_expanded = v.broadcast_axis(X_dim)
                x_i_expanded = x_i.broadcast_axis(V_dim)

                # Rearrange for proper dimension alignment
                v_expanded = v_expanded.rearrange((Pos, V_dim, X_dim))
                x_i_expanded = x_i_expanded.rearrange((Pos, X_dim, V_dim))

                # Transpose the last two dims of x_i_expanded to align with v_expanded
                x_i_expanded = x_i_expanded.rearrange((Pos, V_dim, X_dim))

                # Multiply for outer product
                outer_product = v_expanded * x_i_expanded
                v = self.dropout(outer_product, key=key_dropout)

                # Sum along the X_dim axis to reduce dimensions
                v = hax.sum(v, X_dim)
            else:
                v = self.dropout(v * x_i, key=key_dropout)

            # Apply filtering
            seq_len = Pos.size
            v = self.filter_fn(v, seq_len, key=key_dropout)

        # First x is special - element-wise product with the filtered result
        y = self.activation(v * x[0])

        # Output projection
        y = self.out_proj(y, key=key_dropout)

        return y
