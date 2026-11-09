"""Shared test helpers: import paths, random positions, and a material-count evaluator
that stands in for a trained model (CPU only, nothing to download)."""
import os
import random
import sys

import chess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "Engine")):
    if path not in sys.path:
        sys.path.insert(0, path)

VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# Positions that exercise the encoders' edge cases
EDGE_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",   # en passant capture available
    "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR b KQkq e3 0 3",     # ... for Black
    "4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1",              # en passant square with no capture
    "r3k2r/8/8/8/8/8/8/R3K1R1 w KQkq - 0 1",          # castling right whose rook has moved
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",           # every right usable
    "1k6/1P6/8/8/8/8/6p1/K7 w - - 0 1",               # promotions both ways
    "8/8/8/8/8/8/8/K6k w - - 0 1",                    # bare kings
]


def random_boards(n, seed=0, max_plies=160, keep=0.2):
    """n positions from random playouts, keeping each position along the way with
    probability keep (so several come from each game, at every stage of it)."""
    rng = random.Random(seed)
    boards = []
    while len(boards) < n:
        board = chess.Board()
        while board.ply() < max_plies and len(boards) < n:
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if rng.random() < keep:
                boards.append(board.copy())
    return boards


def material(board):
    """White-perspective material in centipawns."""
    return sum(VALUES[p.piece_type] * (1 if p.color == chess.WHITE else -1) for p in board.piece_map().values())


class MaterialEvaluator:
    """What Bot needs from an evaluator: evaluate(boards) -> White-perspective scores."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, boards):
        self.calls += 1
        return np.array([material(b) for b in boards], dtype=np.float32)


class FailingEvaluator:
    """An evaluator that always raises, like a GPU in a bad state."""

    def evaluate(self, boards):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


def make_bot(depth=2, movetime=None, evaluator=None, **kwargs):
    from bots import Bot
    return Bot("stub", evaluator or MaterialEvaluator(), depth=depth, movetime=movetime, **kwargs)
