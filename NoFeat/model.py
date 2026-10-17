"""
Transformer position evaluator.

66 tokens: 64 squares + 1 castling-rights token + 1 side-to-move token. Each
token is an embedding of its content plus a learned positional embedding. After
the encoder, the tokens are mean-pooled and a small MLP regresses the (scaled)
evaluation from White's perspective.

V4 uses 64 tokens instead: it flips each position to the side to move and folds the
castling rights into the rook tokens (see ChessTransformerV4).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from NoFeat.augment import mirror_tokens
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

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, spatial_bias=False, n_tokens=66):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, spatial_bias, n_tokens)
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


# ---------------------------------------------------------------------------
# V4: V3 on exactly 64 tokens, one per square, so attention fits one 64 x 64 tile.
# The side to move and the castling rights live in the square tokens instead.
# ---------------------------------------------------------------------------

# Model-side square tokens: the data's 0-13, plus a rook that can still castle
CASTLE_ROOK_WHITE, CASTLE_ROOK_BLACK = 14, 15   # after the flip, White is the side to move
NUM_V4_TOKENS = 16
# Per castling bit (columns: white K, white Q, black K, black Q): rook's start square,
# king's start square, rook token, king token, castling-rook token
_CASTLE = torch.tensor([[7, 0, 63, 56],
                        [4, 4, 60, 60],
                        [4, 4, 10, 10],
                        [6, 6, 12, 12],
                        [CASTLE_ROOK_WHITE, CASTLE_ROOK_WHITE, CASTLE_ROOK_BLACK, CASTLE_ROOK_BLACK]])
_CASTLE_ON = {}   # device -> (_CASTLE, arange(4)) there


def canonical_tokens(squares, castling, turn):
    """squares long[B, 64], castling long[B], turn long[B] -> (tokens long[B, 64], flipped bool[B]).

    Rows with Black to move are mirrored (mirror_tokens), so every row has White to
    move; then each castling right turns its rook into token 14 (White's, h1 / a1) or
    15 (Black's, h8 / a8). A right whose rook or king isn't on its start square can
    never be used (python-chess's clean_castling_rights drops it, but eval-DB FENs can
    still carry it), so it is ignored: lossless for every right that matters."""
    flipped = turn == 1
    msq, mca, _ = mirror_tokens(squares, castling, turn)
    squares = torch.where(flipped[:, None], msq, squares)
    castling = torch.where(flipped, mca, castling)
    table = _CASTLE_ON.get(squares.device)
    if table is None:   # copy the tables to the device once, not on every call
        table = _CASTLE_ON[squares.device] = (_CASTLE.to(squares.device), torch.arange(4, device=squares.device))
    (rook_sq, king_sq, rook, king, castle_rook), shifts = table
    rooks = squares[:, rook_sq]                                                   # [B, 4]
    bits = (castling[:, None] >> shifts) & 1                                      # [B, 4]
    usable = (bits == 1) & (rooks == rook) & (squares[:, king_sq] == king)
    return squares.index_copy(1, rook_sq, torch.where(usable, castle_rook, rooks)), flipped


class ChessTransformerV4(nn.Module):
    """V3 blocks on 64 square tokens (canonical_tokens): the position is seen from the
    side to move and the output negated back for flipped rows, so it is still from
    White's perspective, and model(mirror(x)) == -model(x) exactly (colour-swap
    augmentation does nothing). spatial_bias=True: a per-head 64 x 64 table per layer."""

    def __init__(self, d_model=256, n_layers=8, n_heads=16, d_ff=1024, dropout=0.0, spatial_bias=False):
        super().__init__()
        self.config = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           d_ff=d_ff, dropout=dropout, spatial_bias=spatial_bias)

        self.square_emb = nn.Embedding(NUM_V4_TOKENS, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList(EncoderBlock(d_model, n_heads, d_ff, dropout, spatial_bias, n_tokens=64)
                                    for _ in range(n_layers))

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, squares, castling, turn):
        """squares: long[B, 64], castling: long[B], turn: long[B] -> float[B]"""
        tokens, flipped = canonical_tokens(squares, castling, turn)
        x = self.square_emb(tokens) + self.pos_emb    # [B, 64, D]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)                  # [B, D]
        out = self.head(x).squeeze(-1)                # [B], from the side to move's perspective
        return torch.where(flipped, -out, out)        # back to White's


# --arch name -> class; checkpoints store the class name so evaluators rebuild the right one
MODELS = {"v1": ChessTransformerV1, "v2": ChessTransformerV2, "v3": ChessTransformerV3,
          "v4": ChessTransformerV4}
