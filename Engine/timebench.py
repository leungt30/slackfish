"""
Benchmark the timed search's time management on this machine: how long each
iterative-deepening depth takes, and how deep the bot gets for a given time per move.

    uv run python Engine/timebench.py                         # 25 positions to depth 6, ~3-5 min
    uv run python Engine/timebench.py --save times.json       # keep the timings
    uv run python Engine/timebench.py --load times.json --floors 5=4,6=3   # re-simulate, no search
    uv run python Engine/timebench.py --budgets 17,10,6,4 --floors 5=4,6=3 --floors 5=3,6=2.5

A bot on a clock (Engine/lichess.py, the GUI's bot clock) deepens one depth at a time
and only starts the next depth if it's predicted to finish in the time left:
    predicted = last depth's time x max(growth of the same step two depths back, GROWTH floor)
(Bot._next_iteration and GROWTH in Engine/bots.py; an unfinished depth is wasted
time, never a loss on time: the bot plays the deepest finished one.)

1. Measure: each position in Engine/data/timing.fen is searched cold, the way the
   Lichess bot searches, through every depth up to --max-depth with no time limit, and
   the seconds per depth are recorded. Prints the growth from one depth to the next.
2. Simulate: for each time budget (seconds per move; Engine/lichess.py spends about
   1/12 of the clock plus the increment, 1/20 under 3 minutes: 17 s at 3:00 left,
   ~4 s at 1:00 in bullet), replays the bot's decisions on those timings with the
   current GROWTH floors and each --floors alternative. Reports how often each depth
   is reached, the average depth and time used, and how often a started depth would
   be cut off at the deadline.

Growth depends mostly on the search and the model (move ordering, pruning), much less
on the hardware, but run it on each machine the bot plays from: absolute depth per
budget is hardware-specific, and so are the shallow depths' fixed costs.
"""
import argparse
import json
import os
import statistics
import sys
import time

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bots
from bots import load_bot

POSITIONS = os.path.join(HERE, "data", "timing.fen")


def load_positions(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def measure(args):
    import torch
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"Measuring {args.bot} on {device}, depths 1-{args.max_depth}")
    bot = load_bot(args.bot, depth=args.max_depth, device=args.device, movetime=1.0)
    bot.root_scores(chess.Board(), margin=0.0, max_depth=3)   # warm-up: CUDA, compiled kernels
    fens = load_positions(args.positions)[:args.limit]
    out = []
    for fen in fens:
        bot.cache.clear()
        bot.tt.clear()
        bot.pondered.clear()
        bot.history.clear()
        bot.last_search = None
        board = chess.Board(fen)
        # Timed mode (every depth gets the root margin) but no deadline, so no depth is skipped
        bot.root_scores(board, margin=0.0, max_depth=args.max_depth)
        secs = list(bot.last_iteration_secs)
        out.append({"fen": fen, "secs": secs})
        print(f"{chess.popcount(board.occupied):2d} pieces  " + " ".join(f"{t:6.2f}" for t in secs), flush=True)
    return {"bot": args.bot, "device": device, "max_depth": args.max_depth, "positions": out}


def growth_report(data):
    print("\nGrowth from one depth to the next (time of depth d / time of depth d-1):")
    print(f"{'step':>8} {'min':>6} {'median':>7} {'p75':>6} {'p90':>6} {'max':>6}   {'GROWTH floor':>12}")
    for d in range(4, data["max_depth"] + 1):
        r = sorted(p["secs"][d - 1] / p["secs"][d - 2] for p in data["positions"]
                   if len(p["secs"]) >= d and p["secs"][d - 2] > 0)
        if not r:
            continue
        q = statistics.quantiles(r, n=20) if len(r) >= 2 else r * 19
        floor = bots.GROWTH.get(d, 6.0 if d % 2 == 0 else 8.0)
        print(f"{d - 1:>3} -> {d:<2} {r[0]:6.2f} {statistics.median(r):7.2f} {q[14]:6.2f} {q[17]:6.2f} {r[-1]:6.2f}   {floor:>11g}x")


def simulate(secs, budget, growth):
    """Replay the bot's iterative deepening on measured per-depth times: (depth, used, cut)."""
    saved = dict(bots.GROWTH)
    bots.GROWTH.clear()
    bots.GROWTH.update(growth)
    try:
        used, depth, took = 0.0, 0, []
        for d, t in enumerate(secs, start=1):
            if took and budget - used < bots.Bot._next_iteration(took, d):
                return depth, used, False
            if used + t > budget:
                return depth, budget, True   # started, cut off at the deadline
            used += t
            depth = d
            took.append(t)
        return depth, used, False
    finally:
        bots.GROWTH.clear()
        bots.GROWTH.update(saved)


def parse_floors(text):
    return {int(k): float(v) for k, v in (item.split("=") for item in text.split(","))}


def simulation_report(data, budgets, alternatives):
    current = dict(bots.GROWTH)
    configs = [("current", current)] + [(text, {**current, **parse_floors(text)}) for text in alternatives]
    n = len(data["positions"])
    deepest = data["max_depth"]
    print(f"\nSimulated decisions per budget ({n} positions; depths above {deepest} not measured, "
          f"so a position finishing depth {deepest} counts as {deepest}):")
    print(f"{'budget':>7} {'floors':>12} " + " ".join(f"{'>=d' + str(d):>6}" for d in range(4, deepest + 1))
          + f" {'avg depth':>9} {'avg used':>9} {'cut off':>8}")
    for budget in budgets:
        for name, growth in configs:
            res = [simulate(p["secs"], budget, growth) for p in data["positions"]]
            reach = " ".join(f"{sum(r[0] >= d for r in res):>3}/{n:<2}" for d in range(4, deepest + 1))
            print(f"{budget:>6g}s {name:>12} {reach} {statistics.mean(r[0] for r in res):>9.2f} "
                  f"{statistics.mean(r[1] for r in res):>8.1f}s {sum(r[2] for r in res):>4}/{n:<3}")
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot", default="t2-sb-wsd-lichessgames-6.97m", help="bot to time (default: the default bot)")
    p.add_argument("--max-depth", type=int, default=6, help="deepest depth to measure (7 can take a minute per position)")
    p.add_argument("--positions", default=POSITIONS, help="FEN file, one per line (default Engine/data/timing.fen)")
    p.add_argument("--limit", type=int, help="only the first N positions")
    p.add_argument("--device", help="torch device (default: cuda if available)")
    p.add_argument("--budgets", default="17,10,6,4", help="seconds per move to simulate, comma-separated")
    p.add_argument("--floors", action="append", default=[], metavar="D=X,...",
                   help="alternative GROWTH floors to simulate, e.g. 5=4,6=3 (repeatable)")
    p.add_argument("--save", help="write the measured timings to this JSON file")
    p.add_argument("--load", help="skip measuring: simulate timings saved with --save (any machine)")
    args = p.parse_args()

    if args.load:
        with open(args.load) as f:
            data = json.load(f)
        print(f"Timings of {data['bot']} on {data['device']} from {args.load}")
    else:
        start = time.time()
        data = measure(args)
        print(f"({time.time() - start:.0f} s)")
        if args.save:
            with open(args.save, "w") as f:
                json.dump(data, f, indent=1)
            print(f"Saved to {args.save}")
    growth_report(data)
    simulation_report(data, [float(b) for b in args.budgets.split(",")], args.floors)


if __name__ == "__main__":
    main()
