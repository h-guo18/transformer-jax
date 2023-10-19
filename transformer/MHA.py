import math
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, pmap
import einops


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                 )
        self.o_proj = nn.Dense(self.embed_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)  # (b l h*d) -> (b l 3*h*d)

        # Separate Q, K, V from linear output
        qkv = einops.rearrange(
            qkv, "b l (h n d) -> n b h l d", n=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b h l d)

        # Determine value outputs
        d_k = q.shape[-1]
        # (b h l d) @(b h d l) -> (b h l l)
        attn_logits = jnp.einsum("bhid, bhjd->bhij ", q, k)
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
        attention = nn.softmax(attn_logits, axis=-1)  # (b h l l)
        # (b h l l) @ (b h l d) -> (b h l d)
        values = jnp.einsum("bhik,bhkj->bhij", attention, v)
        values = einops.rearrange(values, "b h l d -> b l (h d)")
        o = self.o_proj(values)

        return o, attention


class MultiheadAttention_Linformer(nn.Module):
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)
    k: int = 32

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                 )
        self.o_proj = nn.Dense(self.embed_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)
        self.EF = nn.Dense(self.k,
                           kernel_init=nn.initializers.xavier_uniform(),
                           bias_init=nn.initializers.zeros)  # (l) -> (k)

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)  # (b l h*d) -> (b l 3*h*d)

        # Separate Q, K, V from linear output
        qkv = einops.rearrange(
            qkv, "b l (h n d) -> n b h d l", n=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b h d l)
        k = self.EF(k)  # (b h d k)
        v = self.EF(v)  # (b h d k)
        q = einops.rearrange(q, "b h d l -> b h l d")
        # Determine value outputs
        d_k = q.shape[-1]
        # (b h l d) @(b h d k) -> (b h l k)
        attn_logits = jnp.einsum("bhid, bhdj->bhij ", q, k)
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
        attention = nn.softmax(attn_logits, axis=-1)  # (b h l k)
        # (b h l k) @ (b h d k) -> (b h l d)
        values = jnp.einsum("bhik,bhjk->bhij", attention, v)
        values = einops.rearrange(values, "b h l d -> b l (h d)")
        o = self.o_proj(values)

        return o, attention


# unit test
if __name__ == "__main__":
    main_rng = random.PRNGKey(42)
    # Example features as input
    main_rng, x_rng = random.split(main_rng)
    x = random.normal(x_rng, (2, 2, 8))
    print("Input Shape:", x.shape)
    # Create attention
    mh_attn = MultiheadAttention(embed_dim=8, num_heads=4)
    # Initialize parameters of attention with random key and inputs
    main_rng, init_rng = random.split(main_rng)
    params = mh_attn.init(init_rng, x)['params']
    # Apply attention with parameters on the inputs
    out, attn = mh_attn.apply({'params': params}, x)
    print('Out', out.shape, out, 'Attention', attn.shape, attn)
