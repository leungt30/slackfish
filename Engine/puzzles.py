"""
Puzzle accuracy: our bots (or DeepMind's searchless models) on the 10,000 Lichess puzzles
from DeepMind's searchless chess paper (Ruoss et al. 2024, Table 1), scored the same way
as their src/puzzles.py, so the numbers line up with the paper's.

    uv run python Engine/puzzles.py t2-sg-wsd-lichessgames-8.87m --depth 4
    uv run python Engine/puzzles.py deepmind:270M --limit 1000 --workers 2

A puzzle starts at the end of its game's PGN; the first move of its solution is the
opponent's, then the engine plays every other move. It's solved only if each of those
moves matches the solution, except that any mating move counts (as on Lichess).
Results are broken down by puzzle rating.

The puzzles come from DeepMind's bucket (data/puzzles.csv, 4.7 MB; Lichess puzzles,
CC0). scripts/setup-deepmind.sh downloads them to external/searchless_chess/data/.
Paper, for reference: 9M 88.9%, 136M 94.5%, 270M 95.4%, Stockfish 16 at 1.5 s 100%.
"""
import argparse
import csv
import io
import multiprocessing as mp
import os
import sys
import time

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bots import load_bot
from match import deepmind_command

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(os.environ.get("SEARCHLESS_CHESS", os.path.join(os.path.dirname(HERE), "external", "searchless_chess")),
                           "data", "puzzles.csv")
BANDS = [0, 1000, 1500, 2000, 2500, 3000, 10_000]
_engine = None   # per worker process


def init_worker(args):
    global _engine
    if args.engine.startswith("deepmind:"):
        uci = chess.engine.SimpleEngine.popen_uci(deepmind_command(args.engine.split(":", 1)[1]), timeout=300)
        _engine = lambda board: uci.play(board, chess.engine.Limit(time=1)).move
    else:
        bot = load_bot(args.engine, depth=args.depth, movetime=args.movetime)
        _engine = bot.choose_move


def solve(puzzle):
    """(rating, solved) for one puzzle row."""
    board = chess.pgn.read_game(io.StringIO(puzzle["PGN"])).end().board()
    for i, uci in enumerate(puzzle["Moves"].split()):
        if i % 2 == 1:
            move = _engine(board)
            if move.uci() != uci:
                board.push(move)
                return int(puzzle["Rating"]), board.is_checkmate()
        board.push_uci(uci)
    return int(puzzle["Rating"]), True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("engine", help="bot name from Engine/bots.py, or deepmind:9M|136M|270M[:fp32]")
    p.add_argument("--depth", type=int, default=3, help="search depth for our bots (default 3)")
    p.add_argument("--movetime", type=float, help="seconds per move instead of a fixed depth (our bots)")
    p.add_argument("--limit", type=int, help="only the first N puzzles (default all 10,000)")
    p.add_argument("--csv", default=DEFAULT_CSV, help="puzzles file (default external/searchless_chess/data/puzzles.csv)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"{args.csv} not found: run scripts/setup-deepmind.sh (or pass --csv)")
    with open(args.csv, newline="") as f:
        puzzles = list(csv.DictReader(f))[:args.limit]

    setting = ("" if args.engine.startswith("deepmind:")
               else f" ({args.movetime:g} s/move)" if args.movetime else f" (depth {args.depth})")
    print(f"{args.engine}{setting}: {len(puzzles)} puzzles, {args.workers} workers")
    results, start = [], time.time()
    # spawn: each worker starts clean and opens CUDA itself
    with mp.get_context("spawn").Pool(args.workers, init_worker, (args,)) as pool:
        for i, result in enumerate(pool.imap_unordered(solve, puzzles, chunksize=4), 1):
            results.append(result)
            if i % 500 == 0 or i == len(puzzles):
                print(f"  {i}/{len(puzzles)}: {100 * sum(s for _, s in results) / i:.1f}% "
                      f"({time.time() - start:.0f} s)", flush=True)

    print(f"\n{'Rating':<12}{'Puzzles':>8}{'Solved':>8}")
    for lo, hi in zip(BANDS, BANDS[1:]):
        band = [s for r, s in results if lo <= r < hi]
        if band:
            name = f"<{hi}" if lo == 0 else f"{lo}+" if hi == BANDS[-1] else f"{lo}-{hi - 1}"
            print(f"{name:<12}{len(band):>8}{100 * sum(band) / len(band):>7.1f}%")
    total = sum(s for _, s in results)
    print(f"{'All':<12}{len(results):>8}{100 * total / len(results):>7.1f}%   ({total}/{len(results)})")


if __name__ == "__main__":
    main()
