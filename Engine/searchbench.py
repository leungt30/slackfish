"""
Benchmark the bot's search on fixed test positions, to measure search changes.

    uv run python Engine/searchbench.py                                   # blunders suite, depth 3
    uv run python Engine/searchbench.py --suite wac --depth 2 --limit 50
    uv run python Engine/searchbench.py --suite all --movetime 1 --label baseline --csv bench.csv
    uv run python Engine/searchbench.py --set use_null_move=False --set check_extensions=0 --label no-nmp

Suites:
  blunders  14 decisive mistakes SlackFish made in its Lichess losses (the
            position before its move). Stockfish scores the bot's move against
            Stockfish's preferred move; the move passes if it loses under 10 win%
            (Lichess's win% formula, from the mover's side). "repeat" marks a
            bot that played the original blunder again. Stockfish results are
            cached in Engine/searchbench_cache.json, so reruns are GPU-only.
  wac       Win At Chess, 300 tactical positions (Engine/data/wac.epd, downloaded
            if missing). Solved if the bot's move is one of the bm moves.
            --offset / --limit pick a slice; all 300 at depth 3 take ~5 minutes.

Each position is searched cold (network cache and transposition table cleared).
--set NAME=VALUE sets an attribute on the Bot before searching (ints, floats,
True/False/None, else a string), to toggle search features; a name the Bot
doesn't have yet only gets a warning. The summary is one line per suite, and
--csv appends it to a CSV file so runs can be compared.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bots import load_bot

CACHE = os.path.join(HERE, "searchbench_cache.json")
WAC = os.path.join(HERE, "data", "wac.epd")
WAC_URL = "https://raw.githubusercontent.com/jdart1/arasan-chess/master/tests/wacnew.epd"
PASS_LOSS = 10.0   # win% a blunder-suite move may lose against Stockfish's best
SF_LIMIT = chess.engine.Limit(depth=18)

# (lichess game, move number, fen before SlackFish's move, move it played, Stockfish's best).
# Rebuilt from https://lichess.org/game/export/{id}; every played move's SAN was checked there.
# (D1kLNYqE move 33, Rbe1, was dropped: Stockfish scores it within 1-2% of its best
# move, so it passed whatever the search played. That game's real mistake is move 22.)
BLUNDERS = [
    ("X6TS7w5k", 16, "r2qr1k1/1Qp2p1p/p5p1/3p3n/3n1B2/P1R1PP2/1P3P1P/3K1BR1 w - - 0 16", "Bg5", "Bxc7"),
    ("Ac9ORe94", 23, "r3k1r1/2pn1p1p/p2b4/3p1p2/Np4q1/1P1B2P1/P4P1P/2QR1KR1 w - - 2 23", "Re1+", "Qc2"),
    ("rfcfDpLV", 12, "r1bqnrk1/pp4pp/2n1pp2/3p2P1/3P1Q2/2NB1N2/PPP2P1P/2KR3R b - - 1 12", "e5", "Nd6"),
    ("D1kLNYqE", 22, "r4r1k/pp5p/1q1b1pnp/2pQ4/2P2P1N/P5P1/1P2B2P/R3R1K1 w - - 1 22", "Bd3", "Nf5"),
    ("JklROtBa", 33, "3r2k1/6bp/2RP1p2/p3pp2/3P4/5N2/P4PPP/6K1 b - - 0 33", "f4", "Bf8"),
    ("l8WZH9yr", 13, "r2q1rk1/1bpnppb1/3p1np1/p5Np/NpBPP2P/4BP2/PPPQ2P1/2KR3R w - - 0 13", "Bxf7+", "e5"),
    ("l8WZH9yr", 14, "r2q2k1/1bpnprb1/3p1np1/p5Np/Np1PP2P/4BP2/PPPQ2P1/2KR3R w - - 0 14", "g4", "Nxf7"),
    ("PdAAsxPd", 22, "2r2rk1/1pN2ppp/p4n2/3p4/8/5P1P/PP4P1/2R1R1K1 w - - 1 22", "Re7", "Red1"),
    ("BTWszipI", 12, "r1br2k1/pp3ppp/2nq1n2/3pp3/8/2NBPN1P/PPP1QPP1/3R1RK1 b - - 1 12", "Bxh3", "h6"),
    ("G2GhGMpo", 44, "6k1/5p2/8/3Q1p1p/8/5P1P/5qPK/8 b - - 0 44", "Qc2", "h4"),
    ("8O7rMqxd", 23, "r2r2k1/1b2bp1p/p3p3/q1n1Q1p1/8/2N1P3/PR1NBPPP/2R3K1 b - - 3 23", "Rxd2", "Rac8"),
    # An attack misjudged by the network, not missed by the search: Black is down material
    # but winning by attack at move 26 (+7.6), where depth 3-5 find good moves and depth 6
    # scores them all about the same and plays Qxh3+ (0.00), with pruning on or off. Then it
    # rates a dead-level position +7.4 for itself and skips the perpetual (31...Rf8+).
    ("szQJ0LKM", 26, "3r1r1k/1Q6/6pp/p1pBb3/P3P1P1/7P/1P3P1q/R2NRK2 b - - 6 26", "Qxh3+", "Bg3"),
    ("szQJ0LKM", 29, "1r3r1k/8/6pp/p2Bb3/P1Q1P1P1/7q/1P2KP2/R2NR3 b - - 0 29", "Rxf2+", "Qf3+"),
    ("szQJ0LKM", 31, "1r5k/8/6pp/p2Bb3/P1Q1P1P1/5K2/1P5q/R2NR3 b - - 2 31", "h5", "Rf8+"),
]


def win_pct(cp):
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


class Judge:
    """Stockfish's score (centipawns, mover's side) after a move, cached on disk."""

    def __init__(self):
        self.engine = None
        self.cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    def score(self, board, move):
        key = f"{board.fen()} {move.uci()}"
        if key not in self.cache:
            if self.engine is None:
                self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
                self.engine.configure({"Threads": 6, "Hash": 256})
            after = board.copy()
            after.push(move)
            info = self.engine.analyse(after, SF_LIMIT)
            self.cache[key] = info["score"].pov(board.turn).score(mate_score=100_000)
            with open(CACHE, "w") as f:
                json.dump(self.cache, f, indent=0, sort_keys=True)
        return self.cache[key]

    def close(self):
        if self.engine is not None:
            self.engine.quit()


def blunder_suite(judge):
    for gid, num, fen, played, best in BLUNDERS:
        board = chess.Board(fen)
        played, best = board.parse_san(played), board.parse_san(best)

        def check(move, board=board, played=played, best=best):
            loss = win_pct(judge.score(board, best)) - win_pct(judge.score(board, move))
            note = f"loses {loss:4.1f}%" + ("  repeat" if move == played else "")
            return loss < PASS_LOSS, note
        yield f"{gid}:{num}", board, check


def wac_suite():
    if not os.path.exists(WAC):
        os.makedirs(os.path.dirname(WAC), exist_ok=True)
        urllib.request.urlretrieve(WAC_URL, WAC)
    for line in open(WAC):
        if line.strip():
            board, ops = chess.Board.from_epd(line)
            bm = ops["bm"]
            note = "bm " + " ".join(board.san(m) for m in bm)
            yield ops.get("id", "?"), board, lambda move, bm=bm, note=note: (move in bm, note)


def parse_value(text):
    consts = {"True": True, "False": False, "None": None}
    if text in consts:
        return consts[text]
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def run(name, positions, bot, args):
    print(f"\n== {name} ==")
    passed = total = nodes = 0
    secs, depths = 0.0, []
    for i, (pid, board, check) in enumerate(positions):
        if i < args.offset:
            continue
        if args.limit is not None and total >= args.limit:
            break
        bot.cache.clear()
        bot.tt.clear()
        bot.pondered.clear()
        bot.history.clear()
        bot.last_search = None
        start = time.perf_counter()
        move, _ = bot.search(board)
        dt = time.perf_counter() - start
        ok, note = check(move)
        passed, total, secs, nodes = passed + ok, total + 1, secs + dt, nodes + bot.nodes
        depths.append(bot.last_depth or 0)
        print(f"{pid:<12} {board.san(move):<7} {'pass' if ok else 'FAIL'}  d{bot.last_depth}"
              f"  {dt:6.2f}s  {bot.nodes:>7} nodes  {note}", flush=True)
    n = max(total, 1)
    row = {"suite": name, "label": args.label, "bot": args.bot, "settings": settings(args),
           "passed": passed, "total": total, "secs": round(secs, 2), "mean_secs": round(secs / n, 3),
           "nodes": nodes, "mean_nodes": round(nodes / n), "mean_depth": round(sum(depths) / n, 2)}
    return row


def settings(args):
    limit = f"movetime={args.movetime:g}" if args.movetime is not None else f"depth={args.depth}"
    return " ".join([limit] + args.set)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=["blunders", "wac", "all"], default="blunders")
    p.add_argument("--bot", default="t2-sb-w-lichess-6.97m", help="bot to test (default: the default bot, t2-sb-w-lichess-6.97m)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--depth", type=int, default=3)
    g.add_argument("--movetime", type=float, help="seconds per position instead of a fixed depth")
    p.add_argument("--limit", type=int, help="positions per suite")
    p.add_argument("--offset", type=int, default=0, help="positions to skip at the start of each suite")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="set a Bot attribute before searching (repeatable)")
    p.add_argument("--label", default="", help="name for this run in the summary")
    p.add_argument("--csv", help="append the summary rows to this CSV file")
    args = p.parse_args()

    bot = load_bot(args.bot, depth=args.depth, movetime=args.movetime)
    for item in args.set:
        attr, sep, value = item.partition("=")
        if not sep:
            p.error(f"--set needs NAME=VALUE, got '{item}'")
        if not hasattr(bot, attr):
            print(f"warning: Bot has no attribute '{attr}' yet; setting it anyway", file=sys.stderr)
        setattr(bot, attr, parse_value(value))

    judge = Judge()
    suites = {"blunders": lambda: blunder_suite(judge), "wac": wac_suite}
    try:
        rows = [run(s, suites[s](), bot, args) for s in suites if args.suite in (s, "all")]
    finally:
        judge.close()

    print()
    for r in rows:
        print(f"{r['suite']:<8} {r['label'] or '-':<12} {r['settings']:<24} {r['passed']:>3}/{r['total']:<3}"
              f"  {r['secs']:8.1f}s ({r['mean_secs']:.2f}/pos)  {r['nodes']:>9} nodes ({r['mean_nodes']}/pos)"
              f"  depth {r['mean_depth']}")
    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
