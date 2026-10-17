"""
Measure bot strength: a bot against another bot, or against Stockfish at fixed
strength levels for an Elo estimate.

    uv run python Engine/match.py t2-w-lichess-6.41m t2-w-6.41m                  # 128 games, depth 2
    uv run python Engine/match.py t2-w-lichess-6.41m stockfish:1700,1900,2100,2300,2500 \\
        --depth 3 --openings 12 --movetime 1

Each opening is sampled --plies deep from the gm2001 opening book (so bots that
always play their best move still get varied games) and played once with each
colour. Games still going at move 150 count as draws. Games run in --workers
processes in parallel; on one GPU, about one per CPU core is the useful maximum.

Against Stockfish, its UCI_LimitStrength / UCI_Elo levels (1320-3190) are the
anchors, and the bot's rating is fitted by maximum likelihood over all games.
That scale is calibrated against engine rating lists, not human ratings.
Stockfish needs to be on PATH (or pass --stockfish).

deepmind:9M / deepmind:136M / deepmind:270M is DeepMind's searchless transformer as the
opponent, a fixed outside benchmark (Engine/deepmind_uci.py; set it up once with
scripts/setup-deepmind.sh). It moves after one forward pass, so --movetime doesn't apply;
deepmind:270M:fp32 runs it in full precision (same moves, slower). It's GPU-bound, so use
--workers 2 at most: four DeepMind processes together were no faster than one.

    uv run python Engine/match.py t2-sg-wsd-lichessgames-8.87m deepmind:270M --depth 4 --openings 16 --workers 2

Bot against bot at equal time instead of equal depth (--bot-movetime: both search for that
many seconds per move), for changes that trade evaluation quality for speed; --weights /
--opponent-weights play a model file (best.pt or model.safetensors) under a NoFeat bot's
name without registering it; --gpus spreads the workers over several GPUs, which also
keeps timed searches from slowing each other down on a shared GPU:

    uv run python Engine/match.py t2-sg-wsd-lichessgames-8.87m t2-sg-wsd-lichessgames-8.87m \\
        --weights runs/new/best.pt --bot-movetime 0.5 --openings 128 --workers 16 --gpus 4
"""
import argparse
import math
import multiprocessing as mp
import os
import shutil
import sys
import time

import chess
import chess.engine
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bots import DEFAULT_BOOK, OpeningBook, load_bot
from play import game_over

MAX_MOVES = 150
_bot = _opponent = None   # per worker process


def book_openings(n, plies, seed):
    """n distinct positions reached by sampling the opening book plies deep."""
    book, rng = OpeningBook(DEFAULT_BOOK, max_plies=plies), np.random.default_rng(seed)
    openings, seen = [], set()
    for _ in range(100 * n):
        board = chess.Board()
        while (move := book.pick(board, rng)) is not None:
            board.push(move)
        if board.ply() == plies and board.epd() not in seen:
            seen.add(board.epd())
            openings.append(board)
            if len(openings) == n:
                return openings
    raise SystemExit(f"only found {len(openings)} distinct {plies}-ply book openings")


def init_worker(args):
    global _bot, _opponent
    # spread the workers over --gpus GPUs (each worker process has its own CUDA context)
    worker = mp.current_process()._identity[0] - 1 if mp.current_process()._identity else 0
    device = f"cuda:{worker % args.gpus}" if args.gpus > 1 else None
    timed = dict(movetime=args.bot_movetime) if args.bot_movetime else {}
    _bot = load_bot(args.bot, depth=args.depth, weights=args.weights, device=device, **timed)
    if args.stockfish_levels:
        _opponent = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    elif args.opponent.startswith("deepmind:"):
        _opponent = chess.engine.SimpleEngine.popen_uci(deepmind_command(args.opponent.split(":", 1)[1]),
                                                       timeout=300)   # loading and compiling the model
    else:
        _opponent = load_bot(args.opponent, depth=args.opponent_depth or args.depth, weights=args.opponent_weights,
                             device=device, **timed)


def deepmind_command(spec):
    """The UCI command for DeepMind's model, in the environment scripts/setup-deepmind.sh makes.
    spec is MODEL or MODEL:fp32 (default fp16 with an fp32 fallback, the same moves ~2.4x faster)."""
    model, _, precision = spec.partition(":")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("SEARCHLESS_CHESS", os.path.join(os.path.dirname(here), "external", "searchless_chess"))
    python = os.path.join(root, ".venv", "bin", "python")
    if not os.path.exists(python):
        raise SystemExit(f"{python} not found: run scripts/setup-deepmind.sh (or set SEARCHLESS_CHESS)")
    return [python, os.path.join(here, "deepmind_uci.py"), "--model", model, "--dir", root,
            "--precision", precision or "fp16"]


def play(task):
    """One game. Returns (level or None, bot's score)."""
    opening, bot_white, level, movetime = task
    board = opening.copy()
    if level is not None:
        _opponent.configure({"UCI_LimitStrength": True, "UCI_Elo": level, "Threads": 1, "Hash": 16})
    while not game_over(board) and board.fullmove_number <= MAX_MOVES:
        if (board.turn == chess.WHITE) == bot_white:
            move = _bot.choose_move(board)
        elif level is not None or isinstance(_opponent, chess.engine.SimpleEngine):
            move = _opponent.play(board, chess.engine.Limit(time=movetime), game=(opening.epd(), bot_white, level)).move
        else:
            move = _opponent.choose_move(board)
        board.push(move)
    outcome = board.outcome(claim_draw=game_over(board))
    score = {"1-0": 1.0, "0-1": 0.0}.get(outcome.result() if outcome else "*", 0.5)
    return level, score if bot_white else 1 - score


def elo(p):
    p = min(max(p, 0.005), 0.995)
    return -400 * math.log10(1 / p - 1)


def fit_rating(results):
    """Maximum-likelihood rating against fixed-Elo opponents, with a 95% interval."""
    expected = lambda r, level: 1 / (1 + 10 ** ((level - r) / 400))
    loglik = lambda r: sum(s * math.log(expected(r, l)) + (1 - s) * math.log(1 - expected(r, l)) for l, s in results)
    best = max(range(0, 4001), key=loglik)
    info = sum((math.log(10) / 400) ** 2 * expected(best, l) * (1 - expected(best, l)) for l, _ in results)
    return best, 1.96 / math.sqrt(info)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bot", help="bot name from Engine/bots.py")
    parser.add_argument("opponent", help="another bot, stockfish:ELO[,ELO...], or deepmind:9M|136M|270M[:fp32]")
    parser.add_argument("--depth", type=int, default=2, help="search depth for the bot (default 2)")
    parser.add_argument("--opponent-depth", type=int, help="search depth for an opponent bot (default: --depth)")
    parser.add_argument("--openings", type=int, default=64, help="openings, each played with both colours (default 64)")
    parser.add_argument("--plies", type=int, default=10, help="opening length in plies (default 10)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--movetime", type=float, default=1.0, help="Stockfish seconds per move (default 1)")
    parser.add_argument("--stockfish", default=shutil.which("stockfish") or "stockfish")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--weights", help="model file for the bot instead of its registered one")
    parser.add_argument("--opponent-weights", help="model file for an opponent bot instead of its registered one")
    parser.add_argument("--bot-movetime", type=float,
                        help="both bots search this many seconds per move instead of to a depth (equal time)")
    parser.add_argument("--gpus", type=int, default=1, help="spread the workers over this many GPUs (default 1)")
    parser.add_argument("--hours", type=float,
                        help="stop after this long and report the games finished (games in progress are dropped)")
    args = parser.parse_args()
    args.stockfish_levels = ([int(x) for x in args.opponent.split(":", 1)[1].split(",")]
                             if args.opponent.startswith("stockfish:") else None)

    openings = book_openings(args.openings, args.plies, args.seed)
    levels = args.stockfish_levels or [None]
    tasks = [(o, bot_white, level, args.movetime) for o in openings for level in levels for bot_white in (True, False)]
    limit = f"{args.bot_movetime:g} s/move" if args.bot_movetime else f"depth {args.depth}"
    bot = f"{args.bot} ({args.weights})" if args.weights else args.bot
    opponent = f"{args.opponent} ({args.opponent_weights})" if args.opponent_weights else args.opponent
    print(f"{bot} ({limit}) vs {opponent}: {len(tasks)} games, {args.workers} workers"
          + (f" on {args.gpus} GPUs" if args.gpus > 1 else ""))

    results, deadline = [], time.time() + args.hours * 3600 if args.hours else None
    # spawn: each worker starts clean and opens CUDA itself
    with mp.get_context("spawn").Pool(args.workers, init_worker, (args,)) as pool:
        pending = pool.imap_unordered(play, tasks)
        for i in range(1, len(tasks) + 1):
            try:
                result = pending.next(timeout=None if deadline is None else max(deadline - time.time(), 0.001))
            except mp.TimeoutError:
                print(f"  time limit ({args.hours:g} h): stopping after {len(results)} games", flush=True)
                break
            results.append(result)
            if args.hours or i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} games, score {sum(s for _, s in results):g}", flush=True)
    if not results:
        raise SystemExit("no games finished")

    for level in levels:
        scores = [s for l, s in results if l == level]
        wins, draws = scores.count(1.0), scores.count(0.5)
        name = f"Stockfish {level}" if level else args.opponent
        perf = f"performance {level + elo(np.mean(scores)):.0f}" if level else f"Elo difference {elo(np.mean(scores)):+.0f}"
        print(f"vs {name}: {sum(scores):g}/{len(scores)}  W/D/L {wins}/{draws}/{len(scores) - wins - draws}  {perf}")
    total = sum(s for _, s in results)
    if args.stockfish_levels and total in (0, len(results)):
        side = "below" if total == 0 else "above"
        print(f"Rating (Stockfish scale): {side} the levels tested; try {'lower' if total == 0 else 'higher'} ones")
    elif args.stockfish_levels:
        rating, margin = fit_rating(results)
        print(f"Rating (Stockfish scale): {rating} +- {margin:.0f} (95%)")


if __name__ == "__main__":
    main()
