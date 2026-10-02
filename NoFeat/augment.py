"""
Colour-swap augmentation: mirror the board top-to-bottom, swap piece colours,
swap castling rights and side to move. The result is the same position with the
roles reversed, so its White-perspective target is the negative of the original.
Equivalent to python-chess's board.mirror().
"""
import torch

from NoFeat.encode import EP_TOKEN, NUM_SQUARE_TOKENS

# 0 empty -> 0, white 1-6 -> black 7-12, black 7-12 -> white 1-6, en passant -> en passant
_SWAP_COLOUR = torch.tensor([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, EP_TOKEN])
assert len(_SWAP_COLOUR) == NUM_SQUARE_TOKENS


def mirror_tokens(squares, castling, turn):
    """squares long[B, 64], castling long[B], turn long[B] -> the mirrored triple."""
    # Square index is rank * 8 + file, so flipping the rank axis maps rank r to 7 - r
    flipped = squares.view(-1, 8, 8).flip(1).reshape(-1, 64)
    squares = _SWAP_COLOUR.to(squares.device)[flipped]
    # castling bits: 0 white K, 1 white Q, 2 black K, 3 black Q -> swap the white and black pairs
    castling = ((castling & 3) << 2) | (castling >> 2)
    return squares, castling, 1 - turn
