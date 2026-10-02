"""
FEN -> transformer tokens, with no hand-engineered features.

The board is always encoded as-is from White's point of view (no flipping), and
targets are always White-perspective centipawns.

Square tokens (64), square index a1=0 ... h8=63:
    0      empty
    1-6    white P N B R Q K
    7-12   black P N B R Q K
    13     empty square that can be captured into en passant

Castling token (1), a 4-bit value 0-15:
    bit 0 white kingside, bit 1 white queenside, bit 2 black kingside, bit 3 black queenside

Turn token (1):
    0 white to move, 1 black to move
"""
import numpy as np

NUM_SQUARE_TOKENS = 14
NUM_CASTLING_TOKENS = 16
NUM_TURN_TOKENS = 2
EP_TOKEN = 13

_PIECE = {c: i + 1 for i, c in enumerate("PNBRQK")} | {c: i + 7 for i, c in enumerate("pnbrqk")}


def encode_fen(fen: str):
    """Return (squares uint8[64], castling int, turn int)."""
    placement, turn, castling, ep = fen.split()[:4]
    squares = np.zeros(64, dtype=np.uint8)

    rank = 7
    file = 0
    for c in placement:
        if c == "/":
            rank -= 1
            file = 0
        elif c.isdigit():
            file += int(c)
        else:
            squares[rank * 8 + file] = _PIECE[c]
            file += 1

    if ep != "-":
        squares[(int(ep[1]) - 1) * 8 + ord(ep[0]) - ord("a")] = EP_TOKEN

    castle = (("K" in castling) * 1 | ("Q" in castling) * 2
              | ("k" in castling) * 4 | ("q" in castling) * 8)

    return squares, castle, 0 if turn == "w" else 1


def encode_fens(fens):
    """Batch version: (squares uint8[N,64], castling uint8[N], turn uint8[N])."""
    n = len(fens)
    squares = np.empty((n, 64), dtype=np.uint8)
    castling = np.empty(n, dtype=np.uint8)
    turn = np.empty(n, dtype=np.uint8)
    for i, fen in enumerate(fens):
        squares[i], castling[i], turn[i] = encode_fen(fen)
    return squares, castling, turn
