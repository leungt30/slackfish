"""
Ponder hit rate from the Lichess bot's logs: how often the opponent played a reply the
bot had pondered, and which one by rank, to choose how many replies to ponder
(PONDER_REPLIES in Engine/bots.py).

    uv run python Engine/ponderstats.py                    # logs/lichess-history.out + logs/lichess.out
    uv run python Engine/ponderstats.py --speed bullet
    uv run python Engine/ponderstats.py logs/lichess.out   # one log

Engine/lichess.py logs each searched move's ponder result: "ponder hit #2/5 at d4"
(the opponent played the 2nd likeliest reply; up to 5 were pondered; it was pondered to
depth 4), "ponder miss: #7/5" (ranked 7th, not searched) or "ponder miss: unranked".
Pondering ranks every reply the opponent has (the transposition table's best move,
then by network score) and searches the first PONDER_REPLIES together, a depth at a
time, until the opponent moves.

Games are grouped by speed and by how many replies were pondered (--ponder-replies),
to compare settings. For each rank: how often the opponent played it, the running total ("pondering only
the top K would have caught this share"), how many of those hits were instant (pondered
to the bot's :DEPTH, so played with no thinking) and the median depth pondering reached.

Caveat for "top K": with fewer replies, each gets a bigger share of the opponent's time,
so pondering would reach deeper and more hits would be instant. The table shows what
is lost (hits on ranks above K); the gain needs a run with the smaller setting.
"""
import argparse
import os
import re
import statistics
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = [os.path.join(ROOT, "logs", "lichess-history.out"), os.path.join(ROOT, "logs", "lichess.out")]
GAME = re.compile(r"^\[(\w{8})\] (?:White|Black) vs \S+ \((\w+),")
MOVE = re.compile(r"^\[(\w{8})\] \d+\.+\S+\s+\(depth (\d+), ponder (hit|miss):? (?:#(\d+)/(\d+)(?: at d(\d+))?|unranked), "
                  r"([\d.]+)s of")


def read(paths, speed_filter):
    """Per (speed, replies pondered): hits, misses and depths by rank. A game's setting is
    the largest N on its lines (logs before --ponder-replies show fewer when the opponent
    had fewer replies than that)."""
    speed, moves = {}, defaultdict(list)   # game -> [(kind, rank, n, depth, secs)]
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as f:
            for line in f:
                g = GAME.match(line)
                if g:
                    speed[g.group(1)] = g.group(2)
                    continue
                m = MOVE.match(line)
                if m:
                    game, _, kind, rank, n, depth, secs = m.groups()
                    moves[game].append((kind, rank and int(rank), n and int(n), depth and int(depth), float(secs)))
    stats = defaultdict(lambda: {"hits": Counter(), "instant": Counter(), "depths": defaultdict(list),
                                 "misses": Counter(), "unranked": 0, "games": set()})
    for game, rows in moves.items():
        sp = speed.get(game, "?")
        ns = [n for _, _, n, _, _ in rows if n]
        if (speed_filter and sp != speed_filter) or not ns:
            continue
        s = stats[(sp, max(ns))]
        s["games"].add(game)
        for kind, rank, n, depth, secs in rows:
            if rank is None:
                s["unranked"] += 1
            elif kind == "hit":
                s["hits"][rank] += 1
                s["instant"][rank] += secs < 0.05
                s["depths"][rank].append(depth)
            else:
                s["misses"][rank] += 1
    return stats


def report(name, s):
    hits, misses = s["hits"], s["misses"]
    total = sum(hits.values()) + sum(misses.values()) + s["unranked"]
    if not total:
        return
    speed, searched = name
    n_hit = sum(hits.values())
    depths = [d for ds in s["depths"].values() for d in ds]
    print(f"\n{speed}, {searched} replies pondered: {len(s['games'])} games, {total} moves after pondering, "
          f"median ponder depth on hits {statistics.median(depths) if depths else '-'}: "
          f"hit {n_hit} ({100 * n_hit / total:.0f}%), instant {sum(s['instant'].values())} "
          f"({100 * sum(s['instant'].values()) / total:.0f}%)")
    print(f"  {'rank':>5} {'played':>7} {'share':>6} {'top-K total':>12} {'instant':>8} {'median ponder depth':>20}")
    running = 0
    deepest = max(list(hits) + list(misses) + [0])
    for rank in range(1, min(deepest, 10) + 1):
        played = hits[rank] + misses[rank]
        running += played
        d = s["depths"][rank]
        inst = f"{s['instant'][rank]}/{hits[rank]}" if hits[rank] else "-"
        print(f"  {'#' + str(rank):>5} {played:>7} {100 * played / total:>5.0f}% {100 * running / total:>11.0f}% {inst:>8} "
              f"{(statistics.median(d) if d else '-'):>20}")
    beyond = sum(c for r, c in misses.items() if r > 10) + sum(c for r, c in hits.items() if r > 10)
    if beyond:
        print(f"  {'#11+':>5} {beyond:>7} {100 * beyond / total:>5.0f}%")
    if s["unranked"]:
        print(f"  {'none':>5} {s['unranked']:>7} {100 * s['unranked'] / total:>5.0f}%   (move not among the replies pondering ranked)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="*", help="log files (default: logs/lichess-history.out and logs/lichess.out)")
    p.add_argument("--speed", choices=["ultraBullet", "bullet", "blitz", "rapid", "classical", "correspondence"],
                   help="only games at this speed")
    args = p.parse_args()
    stats = read(args.logs or LOGS, args.speed)
    if not stats:
        raise SystemExit("no moves with ponder ranks in the logs (logged since Engine/lichess.py records them)")
    for name in sorted(stats):   # one table per speed and number of replies pondered
        report(name, stats[name])


if __name__ == "__main__":
    main()
