"""
Reusable SlackFish bots: model loading, position evaluation and search.

    from bots import load_bot
    bot = load_bot("v2", depth=2)
    move = bot.choose_move(board)

Add a new model by writing an encode function and adding an entry to BOTS.
"""
import itertools
import os, sys
from dataclasses import dataclass
from typing import Callable

import chess
import joblib
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "Models", "Training")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Models import SlackFishCNN_V1, SlackFishCNN_V2, SlackFishCNN_V5
import featureEng

MATE_SCORE = 100_000
piece_order = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
piece_type_map = {'p': chess.PAWN, 'n': chess.KNIGHT, 'b': chess.BISHOP,
                  'r': chess.ROOK, 'q': chess.QUEEN, 'k': chess.KING}


def piece_bitboards(board: chess.Board) -> np.ndarray:
    """(12, 8, 8) array, one plane per piece in piece_order."""
    planes = np.zeros((12, 64), dtype=np.float32)
    for i, symbol in enumerate(piece_order):
        color = chess.WHITE if symbol.isupper() else chess.BLACK
        for sq in board.pieces(piece_type_map[symbol.lower()], color):
            planes[i, sq] = 1
    return planes.reshape(12, 8, 8)


def to_2ch(a, b):
    return np.concatenate([np.reshape(a, (1, 8, 8)), np.reshape(b, (1, 8, 8))], axis=0)


# ---------------------------------------------------------------------------
# Encoders: board -> (bitboards, raw scalar features) for each model family
# ---------------------------------------------------------------------------

def encode_v1_v2(board: chess.Board):
    fen = board.fen()
    features = np.array([
        featureEng.piece_count(fen),
        *featureEng.legal_moves_per_side(fen),
        featureEng.player_turn(fen),
        featureEng.en_passant_available(fen),
        featureEng.in_check(fen),
        *featureEng.castling_rights(fen),
        featureEng.pst_score(fen),
    ], dtype=np.float32)
    return piece_bitboards(board), features


def encode_v5(board: chess.Board):
    fen = board.fen()
    planes = np.concatenate([
        piece_bitboards(board),
        to_2ch(*featureEng.isolated_pawns(fen)),
        to_2ch(*featureEng.double_pawns(fen)),
        to_2ch(*featureEng.rook_on_7th_rank(fen)),
        to_2ch(*featureEng.rook_on_semi_open_file(fen)),
        to_2ch(*featureEng.rooks_on_same_file(fen)),
        to_2ch(*featureEng.hanging_pieces_bitboards(fen)),
    ], axis=0)
    features = np.array([
        featureEng.piece_count(fen),
        featureEng.player_turn(fen),
        featureEng.en_passant_available(fen),
        featureEng.in_check(fen),
        *featureEng.castling_rights(fen),
        featureEng.pst_score(fen),
        *featureEng.check_one_move_away(fen),
        *featureEng.legal_moves_per_side(fen),
        *featureEng.is_forking(fen),
        *featureEng.bishop_activity(fen),
        *featureEng.pinned_pieces(fen),
        *featureEng.value_of_hanging_pieces(fen),
        *featureEng.center_control(fen),
        featureEng.is_win(fen),
    ], dtype=np.float32)
    return planes, features


# ---------------------------------------------------------------------------
# Bot registry
# ---------------------------------------------------------------------------

@dataclass
class BotSpec:
    build_model: Callable[[], nn.Module]
    encode: Callable[[chess.Board], tuple]
    weights: str             # relative to repo root
    scaler_y: str            # relative to repo root
    scaler_x: str | None = None  # None = features were not scaled in training

    def make_evaluator(self, weights=None, device=None):
        return Evaluator(self, weights, device)


@dataclass
class NoFeatSpec:
    """Transformer from NoFeat/: the checkpoint carries its own config and target scale."""
    weights: str

    def make_evaluator(self, weights=None, device=None):
        from NoFeat.evaluator import NoFeatEvaluator
        path = weights or os.path.join(ROOT, self.weights)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint at {path}. Train it with `uv run python -m NoFeat.train`")
        return NoFeatEvaluator(path, device)


BOTS = {
    "v1": BotSpec(lambda: SlackFishCNN_V1(11), encode_v1_v2,
                  "Models/Weights/SlackFishCNN_V1_499",
                  "Models/Scalers/scaler_Y.pkl", "Models/Scalers/scaler_X.pkl"),
    "v2": BotSpec(lambda: SlackFishCNN_V2(11), encode_v1_v2,
                  "Models/Weights/SlackFishCNN_V2_449",
                  "Models/Scalers/scaler_Y.pkl", "Models/Scalers/scaler_X.pkl"),
    "v5": BotSpec(lambda: SlackFishCNN_V5(24, 24), encode_v5,
                  "Models/Weights/SlackFishCNN_V5_79",
                  "Models/Scalers/scaler_Y_9M.pkl"),
    "t1-3.25m": NoFeatSpec("Models/Weights/SlackFishV-T1-3.25M/best.pt"),
    "t1-alpha": NoFeatSpec("Models/Weights/SlackFishV-T1-alpha/best.pt"),
    "t2-6.41m": NoFeatSpec("Models/Weights/SlackFishV-T2-6.41M/best.pt"),   # ChessTransformerV2
}


class Evaluator:
    """Wraps a trained model. Scores are centipawns from White's perspective."""

    def __init__(self, spec: BotSpec, weights: str | None = None, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.encode = spec.encode
        self.scaler_x = joblib.load(os.path.join(ROOT, spec.scaler_x)) if spec.scaler_x else None
        self.scaler_y = joblib.load(os.path.join(ROOT, spec.scaler_y))

        path = weights or os.path.join(ROOT, spec.weights)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint at {path}. Train it first or pass weights=... / --white-weights")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model = spec.build_model().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]) -> np.ndarray:
        encoded = [self.encode(b) for b in boards]
        planes = np.stack([e[0] for e in encoded])
        features = np.stack([e[1] for e in encoded])
        if self.scaler_x is not None:
            features = self.scaler_x.transform(features)

        out = self.model(torch.tensor(planes, dtype=torch.float32, device=self.device),
                         torch.tensor(features, dtype=torch.float32, device=self.device))
        return self.scaler_y.inverse_transform(out.cpu().numpy()).reshape(-1)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def terminal_score(board: chess.Board, ply: int):
    """White-perspective score if the game is over, else None. Faster mates score higher."""
    if board.is_checkmate():
        return -(MATE_SCORE - ply) if board.turn == chess.WHITE else MATE_SCORE - ply
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_fifty_moves() or board.is_repetition(3)):
        return 0.0
    return None


def ordered_moves(board: chess.Board):
    """Captures and promotions first, which lets alpha-beta prune more."""
    return sorted(board.legal_moves,
                  key=lambda m: not (board.is_capture(m) or m.promotion))


class Bot:
    """Minimax with alpha-beta pruning over a neural evaluator.

    depth=1 matches the original scripts: score every legal move, pick the best.
    Leaf positions are evaluated in one batch per node, so the model sees
    ~30 positions per forward pass instead of one.

    Network scores are cached by position for the bot's lifetime, so positions
    reached by transposition, or again in the next move's search, skip the
    model. When the cache passes cache_size entries the oldest quarter is dropped.

    temperature=0 always plays the best move. temperature>0 samples a move from
    softmax(score / temperature), with the temperature in centipawns: at 50, a
    move 50 cp worse than another is e^-1 = 0.37x as likely. Pass seed for a
    reproducible sequence of samples.

    To keep sampling cheap, root moves more than sample_margin temperatures
    worse than the best are pruned and given probability 0: each would have had
    at most e^-sample_margin (1.8% at the default 4) of the best move's
    probability. sample_margin=None scores every move exactly.
    """

    def __init__(self, name: str, evaluator: Evaluator, depth: int = 1,
                 cache_size: int = 1_000_000, temperature: float = 0.0, seed: int | None = None,
                 sample_margin: float | None = 4.0):
        self.name = name
        self.evaluator = evaluator
        self.depth = depth
        self.temperature = temperature
        self.sample_margin = sample_margin
        self.rng = np.random.default_rng(seed)
        self.last_distribution = None   # (moves, scores, probs) from the last sampled move
        self.cache_size = cache_size
        self.cache: dict[int, float] = {}
        self.nodes = 0        # positions scored by the model in the last search
        self.cache_hits = 0   # positions answered from the cache in the last search

    def __str__(self):
        temp = f", temperature {self.temperature:g}" if self.temperature > 0 else ""
        return f"SlackFish-{self.name} (depth {self.depth}{temp})"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self.search(board)[0]

    def search(self, board: chess.Board):
        """Return (move, white-perspective score): the best move, or a sampled one if temperature > 0."""
        if self.temperature > 0:
            moves, scores, probs = self.move_distribution(board)
            i = self.rng.choice(len(moves), p=probs)
            return moves[i], float(scores[i])

        self.nodes = self.cache_hits = 0
        board = board.copy()
        maximizing = board.turn == chess.WHITE
        if self.depth <= 1:
            moves, scores = self._score_children(board, ply=1)
            best = int(np.argmax(scores) if maximizing else np.argmin(scores))
            return moves[best], float(scores[best])

        alpha, beta = -float('inf'), float('inf')
        best_move, best_score = None, None
        for move in ordered_moves(board):
            board.push(move)
            score = self._minimax(board, self.depth - 1, alpha, beta, ply=1)
            board.pop()
            if best_score is None or (score > best_score if maximizing else score < best_score):
                best_move, best_score = move, score
            if maximizing:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
        return best_move, best_score

    def root_scores(self, board: chess.Board, margin: float | None = None):
        """White-perspective score of every legal move. With margin=None every score is
        exact. Otherwise only moves within margin cp of the best are scored exactly and
        the rest are NaN: each root move is searched with its window's bound at
        best_so_far - margin, so alpha-beta can cut a move off as soon as it is provably
        that much worse. Depth 1 is a single batch and always exact."""
        self.nodes = self.cache_hits = 0
        board = board.copy()
        if self.depth <= 1:
            return self._score_children(board, ply=1)

        sign = 1 if board.turn == chess.WHITE else -1   # scores from the mover's side = sign * score
        # Search the most promising moves first (one cheap depth-1 batch), so the
        # best score, and with it the cutoff, is found early.
        moves, quick = self._score_children(board, ply=1)
        order = np.argsort(-sign * quick)
        moves = [moves[i] for i in order]
        scores = np.full(len(moves), np.nan)
        best = None   # best score so far, mover's side
        for i, move in enumerate(moves):
            floor = -float('inf') if margin is None or best is None else best - margin
            alpha, beta = (floor, float('inf')) if sign == 1 else (-float('inf'), -floor)
            board.push(move)
            score = sign * self._minimax(board, self.depth - 1, alpha, beta, ply=1)
            board.pop()
            if score <= floor:
                continue   # failed low: an upper bound, so at least margin worse than the best
            scores[i] = sign * score
            best = score if best is None else max(best, score)
        if margin is not None:   # moves that were close when searched but fell behind a later best
            scores[sign * scores < best - margin] = np.nan
        return moves, scores

    def move_distribution(self, board: chess.Board, temperature: float | None = None):
        """Return (moves, white-perspective scores, probabilities) with
        probabilities = softmax(score from the mover's side / temperature)."""
        temperature = temperature or self.temperature or 1.0
        margin = None if self.sample_margin is None else self.sample_margin * temperature
        moves, scores = self.root_scores(board, margin)
        logits = (scores if board.turn == chess.WHITE else -scores) / temperature
        logits = np.where(np.isnan(logits), -np.inf, logits)   # pruned moves get probability 0
        probs = np.exp(logits - logits.max())   # subtract the max so mate scores can't overflow
        probs /= probs.sum()
        self.last_distribution = (moves, scores, probs)
        return moves, scores, probs

    def _minimax(self, board, depth, alpha, beta, ply):
        term = terminal_score(board, ply)
        if term is not None:
            return term
        if depth == 1:
            _, scores = self._score_children(board, ply + 1)
            return float(scores.max() if board.turn == chess.WHITE else scores.min())

        maximizing = board.turn == chess.WHITE
        best = -float('inf') if maximizing else float('inf')
        for move in ordered_moves(board):
            board.push(move)
            score = self._minimax(board, depth - 1, alpha, beta, ply + 1)
            board.pop()
            if maximizing:
                best = max(best, score)
                alpha = max(alpha, best)
            else:
                best = min(best, score)
                beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    def _score_children(self, board, ply):
        """Score every legal move from board; game-ending moves are scored exactly."""
        moves = list(board.legal_moves)
        scores = np.zeros(len(moves))
        to_eval, idx, keys = [], [], []
        for i, move in enumerate(moves):
            board.push(move)
            term = terminal_score(board, ply)
            if term is not None:
                scores[i] = term
            else:
                # Pieces, side to move, castling and legal en passant: everything the
                # models see. Hashed to an int to keep the cache small.
                key = hash(board._transposition_key())
                cached = self.cache.get(key)
                if cached is not None:
                    scores[i] = cached
                    self.cache_hits += 1
                else:
                    to_eval.append(board.copy(stack=False))
                    idx.append(i)
                    keys.append(key)
            board.pop()
        if to_eval:
            values = self.evaluator.evaluate(to_eval)
            scores[idx] = values
            self.nodes += len(to_eval)
            self.cache.update(zip(keys, values.tolist()))
            if len(self.cache) > self.cache_size:
                for key in list(itertools.islice(self.cache, len(self.cache) // 4)):
                    del self.cache[key]
        return moves, scores


_evaluator_cache: dict = {}

def load_bot(name: str, depth: int = 1, weights: str | None = None, device=None,
             temperature: float = 0.0, seed: int | None = None) -> Bot:
    """Load a bot by registry name (see BOTS). Models are cached, so two bots
    with the same weights at different depths share one copy on the GPU.
    temperature > 0 samples moves instead of always playing the best (see Bot)."""
    if name not in BOTS:
        raise ValueError(f"Unknown bot '{name}'. Available: {', '.join(BOTS)}")
    key = (name, weights, device)
    if key not in _evaluator_cache:
        _evaluator_cache[key] = BOTS[name].make_evaluator(weights, device)
    return Bot(name, _evaluator_cache[key], depth, temperature=temperature, seed=seed)
