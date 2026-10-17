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


def pack_squares(squares):
    """uint8[N, 64] square tokens -> uint8[N, 32], two squares per byte (tokens fit
    in 4 bits): byte i holds square 2i in its low nibble and 2i+1 in its high one."""
    return squares[:, 0::2] | (squares[:, 1::2] << 4)


def unpack_squares(packed):
    """Inverse of pack_squares, for numpy arrays or torch tensors."""
    low, high = packed & 15, packed >> 4
    if isinstance(packed, np.ndarray):
        return np.stack((low, high), axis=-1).reshape(len(packed), 64)
    import torch
    return torch.stack((low, high), dim=-1).reshape(packed.shape[0], 64)


_TOKENS = np.arange(1, 13, dtype=np.uint8)[None, :, None]   # white P..K = 1-6, black p..k = 7-12


def encode_boards(boards):
    """encode_fens for chess.Board objects, read from their bitboards instead of building
    and parsing a FEN (~4x faster; what the engine's evaluator uses). Same tokens:
    en passant only when a legal capture exists (as Board.fen() writes it), castling
    rights as FEN reports them (Board.clean_castling_rights)."""
    n = len(boards)
    masks = np.empty((n, 12), dtype=np.uint64)
    castling = np.empty(n, dtype=np.uint8)
    turn = np.empty(n, dtype=np.uint8)
    ep = []
    for i, b in enumerate(boards):
        white, black = b.occupied_co[True], b.occupied_co[False]
        pieces = (b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings)
        masks[i] = [p & white for p in pieces] + [p & black for p in pieces]
        rights = b.clean_castling_rights()
        castling[i] = ((rights >> 7 & 1) | (rights & 1) << 1          # H1 (K), A1 (Q)
                       | (rights >> 63 & 1) << 2 | (rights >> 56 & 1) << 3)   # H8 (k), A8 (q)
        turn[i] = not b.turn
        if b.ep_square is not None and b.has_legal_en_passant():
            ep.append((i, b.ep_square))
    # bit k of a mask is square k: little-endian bytes, least significant bit first
    bits = np.unpackbits(masks.astype("<u8").view(np.uint8).reshape(n, 12, 8), axis=2, bitorder="little")
    squares = (bits * _TOKENS).sum(axis=1, dtype=np.uint8)
    for i, sq in ep:
        squares[i, sq] = EP_TOKEN
    return squares, castling, turn
