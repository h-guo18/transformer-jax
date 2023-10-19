from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, pmap

from MHA import MultiheadAttention, MultiheadAttention_Linformer
from utils import PositionalEncoding


class EncoderBlock(nn.Module):
    # Input dimension is needed here since it is equal to the output dimension (residual connection)
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float
    linformer: bool = False

    def setup(self):
        # Attention layer
        if self.linformer:
            self.self_attn = MultiheadAttention_Linformer(embed_dim=self.input_dim,
                                                          num_heads=self.num_heads)
        else:
            self.self_attn = MultiheadAttention(embed_dim=self.input_dim,
                                                num_heads=self.num_heads)
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim)
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        # Attention part
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        # MLP part
        linear_out = x
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(
                l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float
    linformer: bool = False

    def setup(self):
        self.layers = [EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward,
                                    self.dropout_prob, self.linformer) for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            x = l(x, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = l(x, mask=mask, train=train)
        return attention_maps


class TransformerPredictor(nn.Module):
    model_dim: int                   # Hidden dimensionality to use inside the Transformer
    num_classes: int                 # Number of classes to predict per sequence element
    # Number of heads to use in the Multi-Head Attention blocks
    num_heads: int
    num_layers: int                  # Number of encoder blocks to use
    dropout_prob: float = 0.0        # Dropout to apply inside the model
    input_dropout_prob: float = 0.0  # Dropout to apply on the input features
    linformer: bool = False

    def setup(self):
        # Input dim -> Model dim
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Dense(self.model_dim)
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(self.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.num_layers,
                                              input_dim=self.model_dim,
                                              dim_feedforward=2*self.model_dim,
                                              num_heads=self.num_heads,
                                              dropout_prob=self.dropout_prob,
                                              linformer=self.linformer)
        # Output classifier per sequence lement
        self.output_net = [
            nn.Dense(self.model_dim),
            nn.LayerNorm(),
            nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.num_classes)
        ]

    def __call__(self, x, mask=None, add_positional_encoding=True, train=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
            train - If True, dropout is stochastic
        """
        x = self.input_dropout(x, deterministic=not train)
        x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask, train=train)
        for l in self.output_net:
            x = l(x) if not isinstance(l, nn.Dropout) else l(
                x, deterministic=not train)
        return x

    def get_attention_maps(self, x, mask=None, add_positional_encoding=True, train=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_dropout(x, deterministic=not train)
        x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(
            x, mask=mask, train=train)
        return attention_maps


# unit test
if __name__ == "__main__":
    main_rng = random.PRNGKey(42)
    # Example features as input
    main_rng, x_rng = random.split(main_rng)
    x = random.normal(x_rng, (3, 16, 128))
    # Create Transformer encoder
    transenc = TransformerEncoder(num_layers=5,
                                  input_dim=128,
                                  num_heads=4,
                                  dim_feedforward=256,
                                  dropout_prob=0.15)
    # Initialize parameters of transformer with random key and inputs
    main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
    params = transenc.init(
        {'params': init_rng, 'dropout': dropout_init_rng}, x, train=True)['params']
    # Apply transformer with parameters on the inputs
    # Since dropout is stochastic, we need to pass a rng to the forward
    main_rng, dropout_apply_rng = random.split(main_rng)
    # Instead of passing params and rngs every time to a function call, we can bind them to the module
    binded_mod = transenc.bind({'params': params}, rngs={
                               'dropout': dropout_apply_rng})
    out = binded_mod(x, train=True)
    print('Out', out.shape)
    attn_maps = binded_mod.get_attention_maps(x, train=True)
    print('Attention maps', len(attn_maps), attn_maps[0].shape)
