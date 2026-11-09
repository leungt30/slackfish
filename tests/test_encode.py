"""T1 and T2: the board -> token encoders agree, and mirroring is consistent.

The engine encodes positions from bitboards (NoFeat.encode.encode_boards), the
games data path through prepare_data.board_bits, and the CSV / eval-DB path from FEN
strings (encode_fens). A model only plays as it was trained if all three agree."""
import unittest

import chess
import numpy as np
import torch

from helpers import EDGE_FENS, random_boards
from NoFeat import encode, prepare_data
from NoFeat.augment import mirror_tokens
from NoFeat.model import ChessTransformerV4, canonical_tokens


def as_tensors(encoded):
    squares, castling, turn = encoded
    return (torch.from_numpy(np.asarray(squares)).long(), torch.from_numpy(np.asarray(castling)).long(),
            torch.from_numpy(np.asarray(turn)).long())


class EncoderParity(unittest.TestCase):
    def setUp(self):
        self.boards = random_boards(2000, seed=1) + [chess.Board(f) for f in EDGE_FENS]

    def test_three_encoders_agree(self):
        from_boards = encode.encode_boards(self.boards)
        from_bits = prepare_data.encode_boards([prepare_data.board_bits(b) for b in self.boards])
        from_fens = encode.encode_fens([b.fen() for b in self.boards])
        for name, i in (("squares", 0), ("castling", 1), ("turn", 2)):
            np.testing.assert_array_equal(from_boards[i], from_fens[i], err_msg=f"encode_boards {name}")
            np.testing.assert_array_equal(from_bits[i], from_fens[i], err_msg=f"board_bits {name}")

    def test_en_passant_token_only_with_a_legal_capture(self):
        legal, none = chess.Board(EDGE_FENS[1]), chess.Board(EDGE_FENS[3])
        squares, _, _ = encode.encode_boards([legal, none])
        self.assertEqual(squares[0, chess.F6], encode.EP_TOKEN)
        self.assertNotIn(encode.EP_TOKEN, squares[1])

    @unittest.expectedFailure   # not fixed yet: encode_fen copies a raw FEN's castling field as is (0.46% of eval-DB positions)
    def test_raw_fen_with_unusable_castling_matches_board(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K1R1 w KQkq - 0 1"   # White's h-rook has moved: no K right
        self.assertEqual(encode.encode_fen(fen)[1], encode.encode_boards([chess.Board(fen)])[1][0])

    def test_pack_round_trip(self):
        squares = encode.encode_boards(self.boards)[0]
        packed = torch.from_numpy(encode.pack_squares(squares))
        np.testing.assert_array_equal(encode.unpack_squares(packed).numpy(), squares)


class Symmetry(unittest.TestCase):
    def setUp(self):
        self.boards = random_boards(300, seed=2) + [chess.Board(f) for f in EDGE_FENS]
        self.x = as_tensors(encode.encode_boards(self.boards))

    def test_mirror_tokens_is_the_mirrored_board(self):
        mirrored = mirror_tokens(*self.x)
        expected = as_tensors(encode.encode_boards([b.mirror() for b in self.boards]))
        for got, want in zip(mirrored, expected):
            torch.testing.assert_close(got, want)

    def test_mirror_twice_is_identity(self):
        for got, want in zip(mirror_tokens(*mirror_tokens(*self.x)), self.x):
            torch.testing.assert_close(got, want)

    def test_canonical_tokens_ignore_colour(self):
        tokens, flipped = canonical_tokens(*self.x)
        m_tokens, m_flipped = canonical_tokens(*mirror_tokens(*self.x))
        torch.testing.assert_close(tokens, m_tokens)
        torch.testing.assert_close(flipped, ~m_flipped)

    def test_v4_model_is_antisymmetric(self):
        """model(mirror(x)) == -model(x): what makes colour-swap targets (y = -y) right."""
        torch.manual_seed(0)
        model = ChessTransformerV4(d_model=32, n_layers=2, n_heads=4, d_ff=64, spatial_bias=True,
                                   smolgen=True, smolgen_summary=16, smolgen_head=16).eval()
        with torch.no_grad():
            out, mirrored = model(*self.x), model(*mirror_tokens(*self.x))
        torch.testing.assert_close(mirrored, -out, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
