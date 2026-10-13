"""
Transformer position evaluator.

66 tokens: 64 squares + 1 castling-rights token + 1 side-to-move token. Each
token is an embedding of its content plus a learned positional embedding. After
the encoder, the tokens are mean-pooled and a small MLP regresses the (scaled)
evaluation from White's perspective.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from NoFeat.encode import NUM_SQUARE_TOKENS, NUM_CASTLING_TOKENS, NUM_TURN_TOKENS


class ChessTransformerV1(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.config = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           d_ff=d_ff, dropout=dropout)

        self.square_emb = nn.Embedding(NUM_SQUARE_TOKENS, d_model)
        self.castling_emb = nn.Embedding(NUM_CASTLING_TOKENS, d_model)
        self.turn_emb = nn.Embedding(NUM_TURN_TOKENS, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 66, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, squares, castling, turn):
        """squares: long[B, 64], castling: long[B], turn: long[B] -> float[B]"""
        x = torch.cat([
            self.square_emb(squares),
            self.castling_emb(castling).unsqueeze(1),
            self.turn_emb(turn).unsqueeze(1),
        ], dim=1)
        x = self.encoder(x + self.pos_emb)
        x = self.norm(x).mean(dim=1)
        return self.head(x).squeeze(-1)


class ChessTransformerV2(nn.Module):
    def __init__(self, d_model=256, n_layers=8, n_heads=16, d_ff=1024, dropout=0.1):
        super().__init__()
        self.config = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           d_ff=d_ff, dropout=dropout)

        self.square_emb = nn.Embedding(NUM_SQUARE_TOKENS, d_model)
        self.castling_emb = nn.Embedding(NUM_CASTLING_TOKENS, d_model)
        self.turn_emb = nn.Embedding(NUM_TURN_TOKENS, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 66, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, squares, castling, turn):
        """squares: long[B, 64], castling: long[B], turn: long[B] -> float[B]"""
        x = torch.cat([
            self.square_emb(squares),
            self.castling_emb(castling).unsqueeze(1),
            self.turn_emb(turn).unsqueeze(1),
        ], dim=1)
        x = self.encoder(x + self.pos_emb)
        x = self.norm(x).mean(dim=1)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# V3: the same kind of pre-LayerNorm encoder as V2, written out instead of using
# nn.TransformerEncoderLayer, so every step can be read and changed.
# Shapes: B = batch, L = 66 tokens, D = d_model, H = n_heads, Dh = D / H.
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Every token looks at every token, through n_heads independent heads of
    head_dim = d_model / n_heads numbers (the query, key and value size).

    spatial_bias: give each head a learned [L, L] table added to its attention scores
    before the softmax, so it can attend by board geometry ("same diagonal", "a knight's
    move away") directly. Starts at zero, i.e. identical to plain attention."""

    def __init__(self, d_model, n_heads, dropout=0.0, spatial_bias=False, n_tokens=66):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must divide evenly into {n_heads} heads")
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # W_q, W_k, W_v as one D x 3D matrix
        self.out = nn.Linear(d_model, d_model)       # W_o: mixes the heads back together
        self.dropout = dropout
        # [H, L, L]: bias[h, i, j] is added to the score of token i attending to token j
        self.bias = nn.Parameter(torch.zeros(n_heads, n_tokens, n_tokens)) if spatial_bias else None

    def forward(self, x):
        B, L, D = x.shape
        # [B, L, 3D] -> three tensors of [B, H, L, Dh]: each head gets its own Dh-number slice
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # softmax(q @ k^T / sqrt(Dh)) @ v, per head: the scores are [B, H, L, L]. PyTorch
        # picks the fastest kernel the GPU supports (memory-efficient on an RTX 2080 Ti).
        # With a bias: softmax(q @ k^T / sqrt(Dh) + bias) @ v, the same table for every position.
        mask = None if self.bias is None else self.bias.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.out(out.transpose(1, 2).reshape(B, L, D))   # [B, H, L, Dh] -> [B, L, D]


class FeedForward(nn.Module):
    """Each token on its own: D -> d_ff -> D."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.dropout(F.gelu(self.up(x))))


class EncoderBlock(nn.Module):
    """Pre-LayerNorm transformer block: normalize, transform, add back (residual)."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, spatial_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, spatial_bias)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))   # tokens exchange information
        x = x + self.dropout(self.ff(self.norm2(x)))     # each token is processed on its own
        return x


class ChessTransformerV3(nn.Module):
    """Defaults: 14.4M parameters (width 384, 8 heads of 48, feed-forward 1536, 8 layers).
    spatial_bias=True adds a learned per-head 66 x 66 attention bias to every layer
    (n_layers * n_heads * 4,356 parameters)."""

    def __init__(self, d_model=384, n_layers=8, n_heads=8, d_ff=1536, dropout=0.0, spatial_bias=False):
        super().__init__()
        self.config = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           d_ff=d_ff, dropout=dropout, spatial_bias=spatial_bias)

        # Input: one D-number vector per token, plus where the token sits
        self.square_emb = nn.Embedding(NUM_SQUARE_TOKENS, d_model)
        self.castling_emb = nn.Embedding(NUM_CASTLING_TOKENS, d_model)
        self.turn_emb = nn.Embedding(NUM_TURN_TOKENS, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 66, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList(EncoderBlock(d_model, n_heads, d_ff, dropout, spatial_bias)
                                    for _ in range(n_layers))

        # Output: average the 66 token vectors, then one number (a win-probability logit
        # or a scaled centipawn score, depending on the training target)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, squares, castling, turn):
        """squares: long[B, 64], castling: long[B], turn: long[B] -> float[B]"""
        x = torch.cat([
            self.square_emb(squares),                 # [B, 64, D]
            self.castling_emb(castling).unsqueeze(1), # [B, 1, D]
            self.turn_emb(turn).unsqueeze(1),         # [B, 1, D]
        ], dim=1) + self.pos_emb                      # [B, 66, D]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)                  # [B, D]
        return self.head(x).squeeze(-1)               # [B]


# --arch name -> class; checkpoints store the class name so evaluators rebuild the right one
MODELS = {"v1": ChessTransformerV1, "v2": ChessTransformerV2, "v3": ChessTransformerV3}
