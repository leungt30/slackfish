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
import time
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
# Hugging Face model repo holding exported NoFeat models as <folder>/model.safetensors
# + config.json (see NoFeat/export.py). A missing model is downloaded from here on
# first use. SLACKFISH_HF_REPO overrides it; set it to "" for local files only.
HF_REPO = os.environ.get("SLACKFISH_HF_REPO", "mathemagicianz/slackfish") or None
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

    def available(self) -> bool:
        return os.path.exists(os.path.join(ROOT, self.weights))


@dataclass
class NoFeatSpec:
    """Transformer from NoFeat/, stored in a folder under Models/Weights: the exported
    model.safetensors + config.json (NoFeat.export) if present, else the training
    checkpoint best.pt. If neither is there, it is downloaded from HF_REPO."""
    weights: str   # folder, relative to repo root

    def local_path(self) -> str | None:
        folder = os.path.join(ROOT, self.weights)
        for name in ("model.safetensors", "best.pt"):
            if os.path.exists(os.path.join(folder, name)):
                return os.path.join(folder, name)
        return None

    def available(self) -> bool:
        return self.local_path() is not None or HF_REPO is not None

    def download(self) -> str:
        from huggingface_hub import hf_hub_download
        name = os.path.basename(self.weights)
        for file in ("config.json", "model.safetensors"):
            path = hf_hub_download(HF_REPO, f"{name}/{file}", local_dir=os.path.dirname(os.path.join(ROOT, self.weights)))
        return path

    def make_evaluator(self, weights=None, device=None):
        from NoFeat.evaluator import NoFeatEvaluator
        path = weights or self.local_path()
        if path is None and HF_REPO is not None:
            path = self.download()
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"No model in {os.path.join(ROOT, self.weights)}. Train it with "
                                    f"`uv run python -m NoFeat.train` or set SLACKFISH_HF_REPO to download it")
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
    "t1-3.25m": NoFeatSpec("Models/Weights/SlackFishV-T1-3.25M"),
    "t1-alpha": NoFeatSpec("Models/Weights/SlackFishV-T1-alpha"),
    "t2-6.41m": NoFeatSpec("Models/Weights/SlackFishV-T2-6.41M"),   # ChessTransformerV2
    "t2-w-6.41m": NoFeatSpec("Models/Weights/SlackFishV-T2-W-6.41M"),   # V2, win-probability target
    "t2-w-all-6.41m": NoFeatSpec("Models/Weights/SlackFishV-T2-W-All-6.41M"),   # V2, win-prob, + tactics
    "t2-w-lichess-6.41m": NoFeatSpec("Models/Weights/SlackFishV-T2-W-Lichess-6.41M"),   # V2, win-prob, + 393M Lichess positions, 1 epoch
    "t3-w-lichess-14.4m": NoFeatSpec("Models/Weights/SlackFishV-T3-W-Lichess-14.4M"),   # V3, 14.4M, same data as t2-w-lichess; weaker (experiment record)
    "t2-sb-w-lichess-6.97m": NoFeatSpec("Models/Weights/SlackFishV-T2-SB-W-Lichess-6.97M"),   # V3 at V2 sizes + spatial attention bias, same data as t2-w-lichess
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
    """White-perspective score if the game is over, else None. Faster mates score higher.

    A position that already occurred (in the game or earlier in this line) also
    scores as a draw, not just the third time: the side that's better then
    avoids heading back into it, and the worse side aims for it. Waiting for the
    third occurrence let a won king-and-queen ending drift into threefold
    repetition, because a twice-seen position looked new to the search (and its
    transposition table entry, from before, said "winning")."""
    if board.is_checkmate():
        return -(MATE_SCORE - ply) if board.turn == chess.WHITE else MATE_SCORE - ply
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_fifty_moves() or board.is_repetition(2)):
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
MAX_DEPTH = 64   # iterative deepening limit for a bot on a time budget (movetime)
PONDER_REPLIES = 5   # opponent replies a bot on a time budget ponders on
# Least assumed time ratio of an iteration to the one before, by the depth it reaches
# (measured with the pruning in _search; deeper ones alternate even 6x / odd 8x)
GROWTH = {2: 12.0, 3: 3.0, 4: 6.0}
NULL_WINDOW = 1.0   # centipawns: the width of a zero-window (PVS, null move, LMR) search
NULL_MIN_DEPTH = 2  # null-move pruning from this depth up, reducing by NULL_R (+1 from depth 6)
NULL_R = 2
LMR_MIN_DEPTH = 2   # late move reductions from this depth up, after the first LMR_FULL moves
# (from 2, not 3: Engine/searchbench.py at 9 s/move avoids 10/12 of the Lichess blunders vs 6/12,
# and solves as many WAC tactics, 86/100 at depth 3, as from 3 in half the time)
LMR_FULL = 3
HISTORY_CP = 30.0   # ordering bonus (cp) for the quiet move with the most history; others scaled
# Pruning margins in centipawns (0 turns one off). Depth-1 nodes send all their children to
# the network and were ~94% of all evaluations, so these skip children that can't matter.
FUTILITY_MARGIN = 150   # depth 1: skip quiet moves when this far below what the side needs
RFP_MARGIN = 150        # depth <= RFP_MAX_DEPTH: stop when this much per ply above it (reverse futility)
RFP_MAX_DEPTH = 2
DELTA_MARGIN = 200      # quiescence: skip a capture that can't get within this of alpha


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
        self.cancel = None    # Event that aborts even the kept search (undo, new game)
        self.thread = threading.Thread(target=bot._ponder, args=(board.copy(), self), daemon=True)
        self.thread.start()

    def aborted(self):
        return self.stop.is_set() and (self.keep is None or self.current != self.keep
                                       or (self.cancel is not None and self.cancel.is_set()))

    def finish(self, board: chess.Board | None = None, cancel: threading.Event | None = None):
        """Stop pondering and wait for the thread. board: the position the bot now
        has to move in, or None to abort whatever is running. Without a time budget,
        a search of board already running is left to complete (setting cancel
        aborts it). With one, everything stops at once: the bot's search carries on
        from the deepest pondered result for board instead (see Bot._ponder)."""
        self.keep = None if board is None or self.bot.movetime is not None else hash(board._transposition_key())
        self.cancel = cancel
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

    Time budget: with movetime (seconds), iterative deepening runs until the time
    is up (up to MAX_DEPTH, not self.depth), then plays from the deepest
    iteration that finished. It doesn't start an iteration it isn't predicted to
    finish in the time left (from how much each iteration grew over the last; an
    unfinished one is wasted), and it stops early on a forced mate or a single
    legal move. Pondering then has no depth limit either: it
    deepens the opponent's PONDER_REPLIES likeliest replies together, a depth at a
    time, until they move. If they play one of them and it reached self.depth
    there, the bot plays that pondered move at once, using none of its time.
    Otherwise its search starts from the deepest depth it finished there.
    """

    def __init__(self, name: str, evaluator: Evaluator, depth: int = 1,
                 cache_size: int = 1_000_000, temperature: float = 0.0, seed: int | None = None,
                 sample_margin: float | None = 4.0, quiescence_depth: int = 6,
                 book: OpeningBook | None = None, movetime: float | None = None):
        self.name = name
        self.evaluator = evaluator
        self.depth = depth
        self.movetime = movetime   # seconds per move; None searches to self.depth
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
        # Search techniques (see _search), each switchable for benchmarking
        self.use_pvs = True          # zero-window search for moves after the first, re-search if better
        self.use_null_move = True    # prune when passing still holds the bound
        self.use_lmr = True          # search late quiet moves 1 ply shallower first
        self.use_killers = True      # try quiet moves that cut off at the same ply early
        self.use_history = True      # bonus for quiet moves that often cut off
        self.check_extensions = 2    # extend checking moves by a ply, at most this many per line (0 = off)
        self.null_min_depth = NULL_MIN_DEPTH   # remaining depth from which null move / LMR apply
        self.lmr_min_depth = LMR_MIN_DEPTH
        self.futility_margin = FUTILITY_MARGIN
        self.rfp_margin = RFP_MARGIN
        self.rfp_max_depth = RFP_MAX_DEPTH
        self.delta_margin = DELTA_MARGIN
        self.killers: dict[int, list] = {}   # ply -> up to 2 quiet moves that caused a cutoff
        self.history: dict[tuple, float] = {}   # (from, to) -> cutoff credit (depth squared)
        self._deadline = None   # time.monotonic() when a timed search must stop
        self._budget = None     # the seconds that deadline was set from
        self._can_time_out = False   # False until a timed search has a result to fall back on
        self.last_depth = None  # deepest iteration the last search finished
        self.last_search = None # (moves, scores, depth, move order) of that iteration, to resume from

    def __str__(self):
        temp = f", temperature {self.temperature:g}" if self.temperature > 0 else ""
        quiet = ("" if self.quiescence_depth == 6 else ", no quiescence" if self.quiescence_depth == 0
                 else f", quiescence {self.quiescence_depth}")
        book = f", book {self.book}" if self.book else ""
        limit = "searches by time" if self.movetime is not None else f"depth {self.depth}"
        return f"SlackFish-{self.name} ({limit}{quiet}{temp}{book})"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self.search(board)[0]

    # --- root -----------------------------------------------------------------

    def search(self, board: chess.Board, stop: threading.Event | None = None,
               movetime: float | None = None):
        """Return (move, white-perspective score): the best move, or a sampled one if
        temperature > 0. A book move returns score None. Setting stop (e.g. when
        the bot's clock runs out) raises SearchAborted. movetime: this move's time
        budget in seconds, for a bot with one (default self.movetime)."""
        if self.movetime is not None:
            self._start_clock(self.movetime if movetime is None else movetime)
        self._abort = None if stop is None else stop.is_set
        try:
            return self._search_root(board)
        finally:
            self._abort = None
            self._deadline = self._budget = None

    def _start_clock(self, movetime):
        self._budget = max(0.0, movetime)
        self._deadline = time.monotonic() + self._budget

    def _timed_out(self):
        return (self._can_time_out and self._deadline is not None
                and time.monotonic() >= self._deadline)

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

    def root_scores(self, board: chess.Board, margin: float | None = None,
                    resume: tuple | None = None, max_depth: int | None = None):
        """White-perspective score of every legal move after iterative deepening to
        self.depth, or for as long as the time budget allows (see Bot). margin=None: every score is exact. Otherwise only moves within
        margin cp of the best are scored exactly and the rest are NaN (margin=0 is
        plain alpha-beta: only the best move is exact). Each root move is searched
        with its window's bound at best_so_far - margin, so alpha-beta can cut a
        move off as soon as it is provably that much worse.

        resume: self.last_search from an earlier call on this position, to carry on
        from the next depth (it's the answer if that one runs out of time).
        max_depth: stop there instead (default self.depth, or MAX_DEPTH timed)."""
        self.nodes = self.cache_hits = 0
        self.killers = {}
        self.history = {k: v / 8 for k, v in self.history.items() if v >= 8}   # old credit fades
        board = board.copy()
        moves, statics, terminal = self._score_children(board, ply=1)
        sign = 1 if board.turn == chess.WHITE else -1
        order = list(np.argsort(-sign * statics, kind="stable"))
        timed = self.movetime is not None
        if max_depth is None:
            max_depth = MAX_DEPTH if timed else self.depth
        result, first, self._can_time_out = None, 1, False
        if resume is not None:
            _, scores, depth, order = resume
            result, self.last_search, first = (moves, scores), resume, depth + 1
            self.last_depth, self._can_time_out = depth, True
            if timed and self._settled(moves, scores, sign):
                first = max_depth + 1
        took = []   # seconds each iteration took, to predict the next one
        for depth in range(first, max_depth + 1):
            last = depth == max_depth
            started = time.monotonic()
            try:
                # Timed: any iteration may be the last, so each one gets the margin
                scores, bounds = self._root_pass(board, moves, statics, terminal, order, depth,
                                                 margin if last or timed else 0.0)
            except SearchAborted:
                if not self._timed_out():
                    raise   # stopped from outside: undo, new game, flag or the end of pondering
                break   # out of time: keep the last finished iteration
            order = sorted(order, key=lambda i: -sign * bounds[i])   # best move first next time
            result, self.last_depth, self._can_time_out = (moves, scores), depth, True
            self.last_search = (moves, scores, depth, order)
            took.append(time.monotonic() - started)
            if timed and (self._settled(moves, scores, sign) or (
                    self._deadline is not None and self._deadline - time.monotonic() < self._next_iteration(took, depth + 1))):
                break
        self._can_time_out = False
        return result

    @staticmethod
    def _next_iteration(took, depth):
        """Predicted seconds for iteration `depth`, given the seconds the ones before
        took. Alpha-beta grows unevenly between odd and even depths, so it assumes
        the growth of the same step two depths back, and at least GROWTH (with the
        pruning in _search, 2 -> 3 is ~1-4x, 3 -> 4 ~3-11x, 4 -> 5 ~2-10x)."""
        floor = GROWTH.get(depth, 6.0 if depth % 2 == 0 else 8.0)
        if len(took) >= 3 and took[-3] >= 0.02:
            return took[-1] * max(took[-2] / took[-3], floor)
        return took[-1] * floor

    @staticmethod
    def _settled(moves, scores, sign):
        """Whether deeper search can't change the answer: one legal move or a forced mate either way."""
        return len(moves) == 1 or abs(np.nanmax(sign * scores)) > MATE_BOUND

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
                child = depth - 1 + self._extension(board, 0)
                ext = int(child >= depth)
                if self.use_pvs and margin == 0 and best is not None:
                    # PVS: only ask whether it beats the best so far; search it fully if so
                    value = self._search(board, child, *self._zero_window(sign, alpha, beta), 1, statics[i], ext)
                    if sign * value > best:
                        value = self._search(board, child, alpha, beta, 1, statics[i], ext)
                else:
                    value = self._search(board, child, alpha, beta, 1, statics[i], ext)
                board.pop()
            bounds[i] = value
            if sign * value <= floor and best is not None:
                continue   # failed low: an upper bound, at least margin worse than the best
            scores[i] = value
            best = sign * value if best is None else max(best, sign * value)
        if margin:   # moves that were close when searched but fell behind a later best
            scores[sign * scores < best - margin] = np.nan
        return scores, bounds

    @staticmethod
    def _zero_window(sign, alpha, beta):
        """The zero-width window just past the bound the side to move must beat."""
        return (alpha, alpha + NULL_WINDOW) if sign == 1 else (beta - NULL_WINDOW, beta)

    def _extension(self, board, ext):
        """1 if the move just made gives check and the line has extensions left."""
        return int(ext < self.check_extensions and board.is_check())

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
            self.last_pondered = True
            resume = self.pondered.pop(key)
            if self.movetime is not None and resume[2] < self.depth:   # carry on deeper with this move's budget
                return self.root_scores(board, margin, resume=resume)
            # Untimed, or already searched to self.depth: play it without thinking
            self.nodes = self.cache_hits = 0
            moves, scores, self.last_depth, _ = self.last_search = resume
            return moves, scores
        return self.root_scores(board, margin)

    # --- pondering ------------------------------------------------------------

    def ponder(self, board: chess.Board) -> Ponder:
        """Start thinking on the opponent's time; board has the opponent to move.
        Call finish() on the result before this bot searches again."""
        return Ponder(self, board)

    def _ponder(self, board, handle):
        """Search the bot's reply to the opponent's moves, the opponent's likeliest
        moves first: the transposition table's best move, then by network score
        from the opponent's side. Without a time budget, each reply in turn is
        searched to self.depth. With one, the PONDER_REPLIES likeliest replies are
        deepened together, a depth at a time, with no limit."""
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
            replies = []   # (move, position key) worth searching, likeliest first
            for i in order:
                board.push(moves[i])
                if not board.is_game_over(claim_draw=True) and not (self.book and self.book.covers(board)):
                    replies.append((moves[i], hash(board._transposition_key())))
                board.pop()
            if self.movetime is None:
                for move, key in replies:
                    if handle.stop.is_set():
                        break
                    board.push(move)
                    handle.current = key
                    self.root_scores(board, margin)
                    self.pondered[(key, margin)] = self.last_search
                    handle.current = None
                    board.pop()
            else:
                replies = replies[:PONDER_REPLIES]
                for depth in range(1, MAX_DEPTH + 1):
                    for move, key in replies:
                        board.push(move)
                        self.root_scores(board, margin, resume=self.pondered.get((key, margin)), max_depth=depth)
                        self.pondered[(key, margin)] = self.last_search
                        board.pop()
        except SearchAborted:
            pass
        except Exception as e:
            print(f"pondering failed: {e}")
        finally:
            self._abort = None
            self._deadline = self._budget = None

    # --- tree -----------------------------------------------------------------

    def _search(self, board, depth, alpha, beta, ply, static, ext=0, null_ok=True):
        """White-perspective value of board (not game over; its network score is
        static) searched depth more plies, then quiescence. Fail-soft: a result
        <= alpha is an upper bound, >= beta a lower bound. ext: check extensions
        used on this line; null_ok: False right after a null move.

        On top of alpha-beta with a transposition table and the network's own move
        ordering (every child is scored in one batch, then searched best first):
        - check extensions: a checking move is searched a ply deeper (at most
          check_extensions per line), so checks and the sacrifices behind them
          get followed through instead of stopping at the horizon
        - null move: if the side to move could pass and still not let the other
          side past the bound, cut (not in check, not in pawn endings: zugzwang)
        - ordering: the TT move, then killer moves (quiet moves that cut off at
          this ply), then by network score plus a history bonus for quiet moves
        - PVS: moves after the first get a zero-window search, which only proves
          they're no better; one that is gets a full re-search
        - LMR: late quiet moves are first searched a ply shallower (two when very
          late), and re-searched at full depth only if they look better
        - reverse futility (depth <= rfp_max_depth): so far above the bound that a shallow
          search won't change it, return the network score
        - futility (depth 1): so far below what it needs that a quiet move can't
          make it up, only captures, promotions, checks and the TT move get scored;
          depth-1 nodes are where almost all network evaluations happen"""
        if (self._abort is not None and self._abort()) or self._timed_out():
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

        maximizing = board.turn == chess.WHITE
        sign = 1 if maximizing else -1
        in_check = board.is_check()
        bound_value = beta if maximizing else alpha   # what the side to move needs to reach
        needed = alpha if maximizing else beta        # what it must beat to improve its score
        quiet_position = not in_check and abs(bound_value) < MATE_BOUND and abs(needed) < MATE_BOUND
        if (self.rfp_margin and depth <= self.rfp_max_depth and quiet_position
                and sign * static - self.rfp_margin * depth >= sign * bound_value):
            return static   # reverse futility: so far ahead that a shallow search won't change it
        if (self.use_null_move and null_ok and depth >= self.null_min_depth and not in_check
                and abs(bound_value) < MATE_BOUND and sign * static >= sign * bound_value
                and self._has_pieces(board, board.turn)):
            board.push(chess.Move.null())
            r = NULL_R + (depth >= 6)
            window = (beta - NULL_WINDOW, beta) if maximizing else (alpha, alpha + NULL_WINDOW)
            value = self._search(board, depth - 1 - r, *window, ply + 1, self._static(board), ext, null_ok=False)
            board.pop()
            if sign * value >= sign * bound_value:
                return bound_value   # don't trust a null-move score as exact, or a mate from it

        alpha0, beta0 = alpha, beta
        floor = None   # mover-side bound on the moves futility pruning skipped
        if self.futility_margin and depth == 1 and quiet_position and sign * static + self.futility_margin <= sign * needed:
            # Futility: a quiet move can't gain futility_margin in one ply, so only
            # captures, promotions, checks and the TT move are worth a network call
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion
                     or m == tt_move or board.gives_check(m)]
            floor = sign * static + self.futility_margin
            if not moves:
                return sign * floor   # an upper bound for the side to move, at most what it needed
            moves, statics, terminal = self._score_children(board, ply + 1, moves)
        else:
            moves, statics, terminal = self._score_children(board, ply + 1)
        killers = self.killers.get(ply, ()) if self.use_killers else ()
        hist_max = max(self.history.values(), default=0.0) if self.use_history else 0.0

        def priority(i):
            m = moves[i]
            if m == tt_move:
                return -2e9
            quiet = not board.is_capture(m) and not m.promotion
            if quiet and m in killers:
                return -1e9 + killers.index(m)
            bonus = HISTORY_CP * self.history.get((m.from_square, m.to_square), 0.0) / hist_max if quiet and hist_max else 0.0
            return -sign * statics[i] - bonus
        order = sorted(range(len(moves)), key=priority)

        # Skipped futile moves are worth at most the floor, so the result can't be below it
        best, best_move = (-sign * float('inf') if floor is None else sign * floor), None
        for n, i in enumerate(order):
            m = moves[i]
            if terminal[i]:
                value = statics[i]
            else:
                quiet = not board.is_capture(m) and not m.promotion
                board.push(m)
                e = self._extension(board, ext)
                child = depth - 1 + e
                gives_check = board.is_check()
                reduce = 0
                if (self.use_lmr and n >= LMR_FULL and depth >= self.lmr_min_depth and quiet and not in_check
                        and not gives_check and m not in killers):
                    reduce = 1 + (n >= 3 * LMR_FULL and depth >= 5)
                if n == 0 or not self.use_pvs:
                    value = self._search(board, child - reduce, alpha, beta, ply + 1, statics[i], ext + e)
                    if reduce and (value > alpha if maximizing else value < beta):
                        value = self._search(board, child, alpha, beta, ply + 1, statics[i], ext + e)
                else:
                    zw = self._zero_window(sign, alpha, beta)
                    value = self._search(board, child - reduce, *zw, ply + 1, statics[i], ext + e)
                    beats = value > alpha if maximizing else value < beta
                    if beats and reduce:   # looked better shallow: confirm at full depth
                        value = self._search(board, child, *zw, ply + 1, statics[i], ext + e)
                        beats = value > alpha if maximizing else value < beta
                    if beats and (value < beta if maximizing else value > alpha):   # inside the window: get its exact value
                        value = self._search(board, child, alpha, beta, ply + 1, statics[i], ext + e)
                board.pop()
            if sign * value > sign * best:
                best, best_move = value, m
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                if not board.is_capture(m) and not m.promotion:
                    self._credit(m, ply, depth)
                break

        bound = UPPER if best <= alpha0 else LOWER if best >= beta0 else EXACT
        self._store(self.tt, key, (depth, _to_tt(best, ply), bound, best_move))
        return best

    def _credit(self, move, ply, depth):
        """A quiet move caused a cutoff: remember it as a killer here and in history."""
        if self.use_killers:
            k = self.killers.setdefault(ply, [])
            if move not in k:
                k.insert(0, move)
                del k[2:]
        if self.use_history:
            key = (move.from_square, move.to_square)
            self.history[key] = self.history.get(key, 0.0) + depth * depth

    @staticmethod
    def _has_pieces(board, color):
        """Whether color has a piece besides pawns and king (null move is unsafe without: zugzwang)."""
        return bool(board.occupied_co[color] & ~(board.pawns | board.kings))

    def _static(self, board):
        """Network score of board (cached), for the one position a null move creates."""
        key = hash(board._transposition_key())
        cached = self.cache.get(key)
        if cached is None:
            cached = float(self.evaluator.evaluate([board.copy(stack=False)])[0])
            self.nodes += 1
            self._store(self.cache, key, cached)
        else:
            self.cache_hits += 1
        return cached

    def _quiesce(self, board, alpha, beta, ply, static, qply):
        """Search captures (or every evasion when in check) until the position is quiet."""
        if (self._abort is not None and self._abort()) or self._timed_out():
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
            if self.delta_margin:   # delta pruning: even winning the piece can't reach alpha
                target = sign * (alpha if maximizing else beta)
                moves = [m for m in moves if sign * static + self.delta_margin + 100 * (
                    (PIECE_VALUES[chess.PAWN] if board.is_en_passant(m) else PIECE_VALUES.get(board.piece_type_at(m.to_square), 0))
                    + (8 if m.promotion else 0)) > target]
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
             book: OpeningBook | None = None, movetime: float | None = None) -> Bot:
    """Load a bot by registry name (see BOTS). Models are cached, so two bots
    with the same weights at different depths share one copy on the GPU.
    temperature > 0 samples moves instead of always playing the best (see Bot).
    book: an OpeningBook to play from before searching (see Bot).
    movetime: seconds per move instead of a fixed depth (see Bot)."""
    if name not in BOTS:
        raise ValueError(f"Unknown bot '{name}'. Available: {', '.join(BOTS)}")
    key = (name, weights, device)
    if key not in _evaluator_cache:
        _evaluator_cache[key] = BOTS[name].make_evaluator(weights, device)
    return Bot(name, _evaluator_cache[key], depth, temperature=temperature, seed=seed,
               quiescence_depth=quiescence_depth, book=book, movetime=movetime)
