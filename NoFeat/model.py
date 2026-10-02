"""
Transformer position evaluator.

66 tokens: 64 squares + 1 castling-rights token + 1 side-to-move token. Each
token is an embedding of its content plus a learned positional embedding. After
the encoder, the tokens are mean-pooled and a small MLP regresses the (scaled)
evaluation from White's perspective.
"""
import torch
import torch.nn as nn

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


# --arch name -> class; checkpoints store the class name so evaluators rebuild the right one
MODELS = {"v1": ChessTransformerV1, "v2": ChessTransformerV2}
