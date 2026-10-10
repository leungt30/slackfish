"""
Reusable SlackFish bots: model loading, position evaluation and search.

    from bots import load_bot
    bot = load_bot("v2", depth=2)
    move = bot.choose_move(board)

Add a new model by writing an encode function and adding an entry to BOTS.
"""
import itertools
import os, sys
import threading
from dataclasses import dataclass
from typing import Callable

import chess
import chess.polyglot
import joblib
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BOOK = os.path.join(ROOT, "Models", "Books", "gm2001.bin")
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
    "t2-w-6.41m": NoFeatSpec("Models/Weights/SlackFishV-T2-W-6.41M/best.pt"),   # V2, win-probability target
    "t2-w-all-6.41m": NoFeatSpec("Models/Weights/SlackFishV-T2-W-All-6.41M/best.pt"),   # V2, win-prob, + tactics
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


def quiet_terminal_score(board: chess.Board, ply: int):
    """Cheaper terminal check for quiescence nodes: checkmate and dead positions only
    (captures can't repeat a position or reset into a fifty-move draw)."""
    if board.is_checkmate():
        return -(MATE_SCORE - ply) if board.turn == chess.WHITE else MATE_SCORE - ply
    if board.is_insufficient_material():
        return 0.0
    return None


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
EXACT, LOWER, UPPER = 0, 1, 2   # transposition table bound types
MATE_BOUND = MATE_SCORE - 1000  # scores beyond this are mates, stored relative to the node


def _to_tt(score, ply):
    """Mate scores count plies from the root; store them as plies from this node."""
    return score + ply if score > MATE_BOUND else score - ply if score < -MATE_BOUND else score


def _from_tt(score, ply):
    return score - ply if score > MATE_BOUND else score + ply if score < -MATE_BOUND else score


def good_captures(board: chess.Board):
    """Captures worth searching in quiescence, plus queen promotions. A capture is
    skipped when it gives up a more valuable piece on a defended square (a cheap
    stand-in for static exchange evaluation)."""
    them = not board.turn
    moves = []
    for m in board.generate_legal_captures():
        if m.promotion and m.promotion != chess.QUEEN:
            continue
        victim = chess.PAWN if board.is_en_passant(m) else board.piece_type_at(m.to_square)
        attacker = board.piece_type_at(m.from_square)
        if PIECE_VALUES[victim] >= PIECE_VALUES[attacker] or not board.is_attacked_by(them, m.to_square):
            moves.append(m)
    pawns = board.pawns & board.occupied_co[board.turn] & (chess.BB_RANK_7 if board.turn else chess.BB_RANK_2)
    for m in board.generate_legal_moves(from_mask=pawns, to_mask=~board.occupied):
        if m.promotion == chess.QUEEN:
            moves.append(m)
    return moves


class OpeningBook:
    """A Polyglot (.bin) opening book. pick() returns a book move for the
    position, sampled in proportion to the book's weights (how often it was
    played), or None once the game is out of book or past max_plies.
    Moves with under min_share of the position's total weight are ignored, so
    one-off oddities in the source games aren't played."""

    def __init__(self, path: str = DEFAULT_BOOK, max_plies: int = 16, min_share: float = 0.02):
        self.path = path
        self.max_plies = max_plies
        self.min_share = min_share
        self.reader = chess.polyglot.open_reader(path)

    def __str__(self):
        return os.path.basename(self.path)

    def _entries(self, board: chess.Board):
        if board.ply() >= self.max_plies:
            return []
        entries = list(self.reader.find_all(board))
        total = sum(e.weight for e in entries)
        return [e for e in entries if e.weight >= self.min_share * total and e.weight > 0]

    def covers(self, board: chess.Board) -> bool:
        """Whether pick() would return a book move here."""
        return bool(self._entries(board))

    def pick(self, board: chess.Board, rng: np.random.Generator) -> chess.Move | None:
        entries = self._entries(board)
        if not entries:
            return None
        weights = np.array([e.weight for e in entries], dtype=float)
        return entries[rng.choice(len(entries), p=weights / weights.sum())].move


class SearchAborted(Exception):
    """Raised inside a search that was told to stop (see Bot.search and Ponder)."""


class Ponder:
    """The bot thinking on its opponent's time, on a background thread (see Bot.ponder).

    finish(board) stops it once the opponent has moved: a ponder search already
    running on board (the move it predicted) is left to complete, any other is
    aborted. Afterwards the bot's search(board) uses the pondered result if there
    is one, and its caches are warm either way. The bot must not search on
    another thread until finish() returns."""

    def __init__(self, bot, board: chess.Board):
        self.bot = bot
        self.stop = threading.Event()
        self.keep = None      # position whose search may finish after stop
        self.current = None   # position being searched now
        self.thread = threading.Thread(target=bot._ponder, args=(board.copy(), self), daemon=True)
        self.thread.start()

    def aborted(self):
        return self.stop.is_set() and self.current != self.keep

    def finish(self, board: chess.Board | None = None):
        """Stop pondering and wait for the thread. board: the position the bot now
        has to move in, or None to abort whatever is running."""
        self.keep = None if board is None else hash(board._transposition_key())
        self.stop.set()
        self.thread.join()


class Bot:
    """Alpha-beta search over a neural evaluator.

    Search: iterative deepening (depth 1, 2, ... self.depth) with a transposition
    table, then quiescence search at the leaves. Every node scores all its children
    in one batched forward pass and searches them best-first by that score, which
    orders moves well enough for alpha-beta to prune most of the tree. The
    transposition table keeps each position's score, bound type and best move, so
    repeated positions and the next iteration (or next move) start from the best
    move found so far.

    Quiescence: past the depth limit, only captures and queen promotions are
    searched (checks get every evasion), for up to quiescence_depth plies. The side
    to move can always "stand pat" on the network score, so a quiet position costs
    nothing extra, but the search no longer stops halfway through an exchange.
    quiescence_depth=0 turns it off (depth 1 then just picks the best-scoring move).

    Network scores are cached by position for the bot's lifetime. Both caches drop
    their oldest quarter when they pass cache_size entries.

    temperature=0 always plays the best move. temperature>0 samples a move from
    softmax(score / temperature), with the temperature in centipawns: at 50, a
    move 50 cp worse than another is e^-1 = 0.37x as likely. Pass seed for a
    reproducible sequence of samples.

    To keep sampling cheap, root moves more than sample_margin temperatures
    worse than the best are pruned and given probability 0: each would have had
    at most e^-sample_margin (1.8% at the default 4) of the best move's
    probability. sample_margin=None scores every move exactly.

    book: an OpeningBook. While the position is in it, the bot plays a book move
    (sampled by the book's weights, from the same seeded rng) instead of
    searching, and search() returns None as the score.

    Pondering: ponder(board), with the opponent to move in board, searches the
    opponent's replies on a background thread, likeliest first, until stopped.
    Each finished reply's root scores are kept, so if the opponent plays one of
    them, search() answers without searching again. Pondering never touches the
    rng, so it doesn't change which moves get sampled.
    """

    def __init__(self, name: str, evaluator: Evaluator, depth: int = 1,
                 cache_size: int = 1_000_000, temperature: float = 0.0, seed: int | None = None,
                 sample_margin: float | None = 4.0, quiescence_depth: int = 6,
                 book: OpeningBook | None = None):
        self.name = name
        self.evaluator = evaluator
        self.depth = depth
        self.temperature = temperature
        self.sample_margin = sample_margin
        self.quiescence_depth = quiescence_depth
        self.book = book
        self.last_book = False   # whether the last search() played a book move
        self.rng = np.random.default_rng(seed)
        self.last_distribution = None   # (moves, scores, probs) from the last sampled move
        self.cache_size = cache_size
        self.cache: dict[int, float] = {}   # position -> network score
        self.tt: dict[int, tuple] = {}      # position -> (depth, score, bound, best move)
        self.nodes = 0        # positions scored by the model in the last search
        self.cache_hits = 0   # positions answered from the cache in the last search
        self.pondered: dict[tuple, tuple] = {}   # (position, margin) -> (moves, scores) from pondering
        self.last_pondered = False   # whether the last search() used a pondered result
        self._abort = None    # callable checked at every node; True stops the search

    def __str__(self):
        temp = f", temperature {self.temperature:g}" if self.temperature > 0 else ""
        quiet = ("" if self.quiescence_depth == 6 else ", no quiescence" if self.quiescence_depth == 0
                 else f", quiescence {self.quiescence_depth}")
        book = f", book {self.book}" if self.book else ""
        return f"SlackFish-{self.name} (depth {self.depth}{quiet}{temp}{book})"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self.search(board)[0]

    # --- root -----------------------------------------------------------------

    def search(self, board: chess.Board, stop: threading.Event | None = None):
        """Return (move, white-perspective score): the best move, or a sampled one if
        temperature > 0. A book move returns score None. Setting stop (e.g. when
        the bot's clock runs out) raises SearchAborted."""
        if stop is None:
            return self._search_root(board)
        self._abort = stop.is_set
        try:
            return self._search_root(board)
        finally:
            self._abort = None

    def _search_root(self, board):
        self.last_book = self.last_pondered = False
        if self.book is not None:
            move = self.book.pick(board, self.rng)
            if move is not None:
                self.nodes = self.cache_hits = 0
                self.last_distribution = None
                self.last_book = True
                return move, None
        if self.temperature > 0:
            moves, scores, probs = self.move_distribution(board)
            i = self.rng.choice(len(moves), p=probs)
            return moves[i], float(scores[i])
        moves, scores = self._root_scores_or_pondered(board, margin=0.0)
        sign = 1 if board.turn == chess.WHITE else -1
        best = int(np.nanargmax(sign * scores))
        return moves[best], float(scores[best])

    def root_scores(self, board: chess.Board, margin: float | None = None):
        """White-perspective score of every legal move after iterative deepening to
        self.depth. margin=None: every score is exact. Otherwise only moves within
        margin cp of the best are scored exactly and the rest are NaN (margin=0 is
        plain alpha-beta: only the best move is exact). Each root move is searched
        with its window's bound at best_so_far - margin, so alpha-beta can cut a
        move off as soon as it is provably that much worse."""
        self.nodes = self.cache_hits = 0
        board = board.copy()
        moves, statics, terminal = self._score_children(board, ply=1)
        sign = 1 if board.turn == chess.WHITE else -1
        order = list(np.argsort(-sign * statics, kind="stable"))
        for depth in range(1, self.depth + 1):
            last = depth == self.depth
            scores, bounds = self._root_pass(board, moves, statics, terminal, order, depth,
                                             margin if last else 0.0)
            order = sorted(order, key=lambda i: -sign * bounds[i])   # best move first next time
        return moves, scores

    def _root_pass(self, board, moves, statics, terminal, order, depth, margin):
        sign = 1 if board.turn == chess.WHITE else -1
        scores = np.full(len(moves), np.nan)
        bounds = np.array(statics, dtype=float)
        best = None   # best score so far, mover's side
        for i in order:
            floor = -float('inf') if margin is None or best is None else best - margin
            if terminal[i]:
                value = statics[i]
            else:
                alpha, beta = (floor, float('inf')) if sign == 1 else (-float('inf'), -floor)
                board.push(moves[i])
                value = self._search(board, depth - 1, alpha, beta, 1, statics[i])
                board.pop()
            bounds[i] = value
            if sign * value <= floor and best is not None:
                continue   # failed low: an upper bound, at least margin worse than the best
            scores[i] = value
            best = sign * value if best is None else max(best, sign * value)
        if margin:   # moves that were close when searched but fell behind a later best
            scores[sign * scores < best - margin] = np.nan
        return scores, bounds

    def move_distribution(self, board: chess.Board, temperature: float | None = None):
        """Return (moves, white-perspective scores, probabilities) with
        probabilities = softmax(score from the mover's side / temperature)."""
        temperature = temperature or self.temperature or 1.0
        moves, scores = self._root_scores_or_pondered(board, self._margin(temperature))
        logits = (scores if board.turn == chess.WHITE else -scores) / temperature
        logits = np.where(np.isnan(logits), -np.inf, logits)   # pruned moves get probability 0
        probs = np.exp(logits - logits.max())   # subtract the max so mate scores can't overflow
        probs /= probs.sum()
        self.last_distribution = (moves, scores, probs)
        return moves, scores, probs

    def _margin(self, temperature):
        """The root_scores margin search() uses at this temperature."""
        if temperature <= 0:
            return 0.0
        return None if self.sample_margin is None else self.sample_margin * temperature

    def _root_scores_or_pondered(self, board, margin):
        key = (hash(board._transposition_key()), margin)
        if key in self.pondered:
            self.nodes = self.cache_hits = 0
            self.last_pondered = True
            return self.pondered.pop(key)
        return self.root_scores(board, margin)

    # --- pondering ------------------------------------------------------------

    def ponder(self, board: chess.Board) -> Ponder:
        """Start thinking on the opponent's time; board has the opponent to move.
        Call finish() on the result before this bot searches again."""
        return Ponder(self, board)

    def _ponder(self, board, handle):
        """Search the bot's reply to each of the opponent's moves, the opponent's
        likeliest moves first: the transposition table's best move, then by
        network score from the opponent's side."""
        self.pondered = {}
        self._abort = handle.aborted
        margin = self._margin(self.temperature)
        try:
            moves, statics, _ = self._score_children(board, ply=1)
            sign = 1 if board.turn == chess.WHITE else -1
            order = sorted(range(len(moves)), key=lambda i: -sign * statics[i])
            entry = self.tt.get(hash(board._transposition_key()))
            if entry is not None and entry[3] in moves:
                first = moves.index(entry[3])
                order.remove(first)
                order.insert(0, first)
            for i in order:
                if handle.stop.is_set():
                    break
                board.push(moves[i])
                if not board.is_game_over(claim_draw=True) and not (self.book and self.book.covers(board)):
                    key = hash(board._transposition_key())
                    handle.current = key
                    self.pondered[(key, margin)] = self.root_scores(board, margin)
                    handle.current = None
                board.pop()
        except SearchAborted:
            pass
        except Exception as e:
            print(f"pondering failed: {e}")
        finally:
            self._abort = None

    # --- tree -----------------------------------------------------------------

    def _search(self, board, depth, alpha, beta, ply, static):
        """White-perspective value of board (not game over; its network score is
        static) searched depth more plies, then quiescence. Fail-soft: a result
        <= alpha is an upper bound, >= beta a lower bound."""
        if self._abort is not None and self._abort():
            raise SearchAborted
        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply, static, 0)

        key = hash(board._transposition_key())
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            entry_depth, value, bound, tt_move = entry
            value = _from_tt(value, ply)
            if entry_depth >= depth and (bound == EXACT or (bound == LOWER and value >= beta)
                                         or (bound == UPPER and value <= alpha)):
                return value

        alpha0, beta0 = alpha, beta
        maximizing = board.turn == chess.WHITE
        sign = 1 if maximizing else -1
        moves, statics, terminal = self._score_children(board, ply + 1)
        order = sorted(range(len(moves)), key=lambda i: -sign * statics[i])
        if tt_move is not None and tt_move in moves:
            first = moves.index(tt_move)
            order.remove(first)
            order.insert(0, first)

        best, best_move = -sign * float('inf'), None
        for i in order:
            if terminal[i]:
                value = statics[i]
            else:
                board.push(moves[i])
                value = self._search(board, depth - 1, alpha, beta, ply + 1, statics[i])
                board.pop()
            if sign * value > sign * best:
                best, best_move = value, moves[i]
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                break

        bound = UPPER if best <= alpha0 else LOWER if best >= beta0 else EXACT
        self._store(self.tt, key, (depth, _to_tt(best, ply), bound, best_move))
        return best

    def _quiesce(self, board, alpha, beta, ply, static, qply):
        """Search captures (or every evasion when in check) until the position is quiet."""
        if self._abort is not None and self._abort():
            raise SearchAborted
        maximizing = board.turn == chess.WHITE
        sign = 1 if maximizing else -1
        in_check = board.is_check()
        if qply >= self.quiescence_depth:
            return static
        if in_check:   # no standing pat in check: every evasion is searched
            best = -sign * float('inf')
            moves = list(board.legal_moves)   # not empty: the caller caught checkmate
        else:
            best = static   # stand pat: the side to move can decline every capture
            if maximizing:
                if static >= beta:
                    return static
                alpha = max(alpha, static)
            else:
                if static <= alpha:
                    return static
                beta = min(beta, static)
            moves = good_captures(board)
            if not moves:
                return static

        moves, statics, terminal = self._score_children(board, ply + 1, moves, quiet=True)
        for i in sorted(range(len(moves)), key=lambda i: -sign * statics[i]):
            if terminal[i]:
                value = statics[i]
            else:
                board.push(moves[i])
                value = self._quiesce(board, alpha, beta, ply + 1, statics[i], qply + 1)
                board.pop()
            if sign * value > sign * best:
                best = value
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                break
        return best

    # --- evaluation -----------------------------------------------------------

    def _score_children(self, board, ply, moves=None, quiet=False):
        """Score each move from board (default: every legal move) in one batch.
        Returns (moves, white-perspective scores, game-over mask); game-ending moves
        are scored exactly. quiet=True uses the cheaper quiescence terminal check."""
        moves = list(board.legal_moves) if moves is None else moves
        scores = np.zeros(len(moves))
        terminal = np.zeros(len(moves), dtype=bool)
        check_terminal = quiet_terminal_score if quiet else terminal_score
        to_eval, idx, keys = [], [], []
        for i, move in enumerate(moves):
            board.push(move)
            term = check_terminal(board, ply)
            if term is not None:
                scores[i], terminal[i] = term, True
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
            self._trim(self.cache)
        return moves, scores, terminal

    def _store(self, table, key, value):
        table[key] = value
        self._trim(table)

    def _trim(self, table):
        if len(table) > self.cache_size:
            for key in list(itertools.islice(table, len(table) // 4)):
                del table[key]


_evaluator_cache: dict = {}

def load_bot(name: str, depth: int = 1, weights: str | None = None, device=None,
             temperature: float = 0.0, seed: int | None = None, quiescence_depth: int = 6,
             book: OpeningBook | None = None) -> Bot:
    """Load a bot by registry name (see BOTS). Models are cached, so two bots
    with the same weights at different depths share one copy on the GPU.
    temperature > 0 samples moves instead of always playing the best (see Bot).
    book: an OpeningBook to play from before searching (see Bot)."""
    if name not in BOTS:
        raise ValueError(f"Unknown bot '{name}'. Available: {', '.join(BOTS)}")
    key = (name, weights, device)
    if key not in _evaluator_cache:
        _evaluator_cache[key] = BOTS[name].make_evaluator(weights, device)
    return Bot(name, _evaluator_cache[key], depth, temperature=temperature, seed=seed,
               quiescence_depth=quiescence_depth, book=book)
