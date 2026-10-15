"""
Ponder hit rate from the Lichess bot's logs: how often the opponent played a reply the
bot had pondered, and which one by rank, to choose how many replies to ponder
(PONDER_REPLIES in Engine/bots.py).

    uv run python Engine/ponderstats.py                    # logs/lichess-history.out + logs/lichess.out
    uv run python Engine/ponderstats.py --speed bullet
    uv run python Engine/ponderstats.py logs/lichess.out   # one log

Engine/lichess.py logs each searched move's ponder result: "ponder hit #2/5 at d4"
(the opponent played the 2nd likeliest reply; 5 were searched; it was pondered to
depth 4), "ponder miss: #7/5" (ranked 7th, not searched) or "ponder miss: unranked".
Pondering ranks every reply the opponent has (the transposition table's best move,
then by network score) and searches the first PONDER_REPLIES together, a depth at a
time, until the opponent moves.

For each rank: how often the opponent played it, the running total ("pondering only
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
    speed = {}
    stats = defaultdict(lambda: {"hits": Counter(), "instant": Counter(), "depths": defaultdict(list),
                                 "misses": Counter(), "unranked": 0, "searched": Counter(), "games": set()})
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
                if not m:
                    continue
                game, _, kind, rank, searched, depth, secs = m.groups()
                sp = speed.get(game, "?")
                if speed_filter and sp != speed_filter:
                    continue
                s = stats[sp]
                s["games"].add(game)
                if rank is None:
                    s["unranked"] += 1
                    continue
                rank, searched = int(rank), int(searched)
                s["searched"][searched] += 1
                if kind == "hit":
                    s["hits"][rank] += 1
                    s["instant"][rank] += float(secs) < 0.05
                    s["depths"][rank].append(int(depth))
                else:
                    s["misses"][rank] += 1
    return stats


def report(name, s):
    hits, misses = s["hits"], s["misses"]
    total = sum(hits.values()) + sum(misses.values()) + s["unranked"]
    if not total:
        return
    searched = s["searched"].most_common(1)[0][0] if s["searched"] else "?"
    n_hit = sum(hits.values())
    print(f"\n{name}: {len(s['games'])} games, {total} moves after pondering ({searched} replies searched): "
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
    for name in sorted(stats):
        report(name, stats[name])


if __name__ == "__main__":
    main()
