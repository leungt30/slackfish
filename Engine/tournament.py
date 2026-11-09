"""
Round-robin between several bots on a clock, with one combined rating fit.

    uv run python Engine/tournament.py t3-sg64-muon-lichessgames-16.0m t2-w-6.41m t1-3.25m \\
        deepmind:270M --clock 1+1 --workers 4 --out logs/tournament/bullet.jsonl

Runs until stopped (Ctrl-C, or --games N). Each task is one book opening played twice
between a pair, once with each colour. Pairs are picked by how much a game between them
tells us: weight p(1-p) under the current ratings, so lopsided pairs still play now and
then (keeping the fit connected) but most games go to close ones. Every game is appended
to --out as a JSON line, so a restart with the same --out carries on from those results.

Players are bots, deepmind:9M|136M|270M, or stockfish:ELO (Stockfish at a UCI_Elo level,
1320-3190, one thread, managing its own clock). --deprioritize BOTS cuts the weight of
pairs where both bots are in that list to a tenth, e.g. weak bots that are close to each
other but tell us nothing about the rest.

Ratings are a maximum-likelihood Elo fit over all games (draws count half), anchored
so that --anchor (default deepmind:270M if playing, else the first bot) is 0, with a
weak prior pulling each rating towards the anchor so 100% scores stay finite. With
Stockfish levels playing, a second column puts the ratings on Stockfish's scale: shifted
so the levels' fitted ratings match their UCI_Elo on average (that scale is calibrated
at longer time controls, so at bullet it's approximate). The table is printed every
--report games and written to <out>.standings.txt.

    uv run python Engine/tournament.py --standings logs/tournament/bullet.jsonl   # just the table
"""
import argparse
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time

import chess
import chess.engine
import chess.pgn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bots import load_bot
from clock import parse_clock, play_clocked
from match import MAX_MOVES, book_openings, deepmind_command

RESULT = "RESULT "   # prefix of the result line a game process prints
PRIOR_SD = 1000   # Elo: the weak prior on each rating's distance from the anchor
_engines = {}     # in a game process: UCI engines by name


def player(name):
    """A bot, or a UCI engine (started now, before the clock)."""
    if name.startswith("stockfish:"):
        if name not in _engines:
            engine = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "stockfish")
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(name.split(":", 1)[1]),
                              "Threads": 1, "Hash": 16, "Move Overhead": 50})
            _engines[name] = engine
        return _engines[name]
    if name.startswith("deepmind:"):
        if name not in _engines:
            engine = chess.engine.SimpleEngine.popen_uci(deepmind_command(name.split(":", 1)[1]), timeout=300)
            engine.ping()   # isready loads and compiles the model: do it now, not on the first move's clock
            _engines[name] = engine
        return _engines[name]
    return load_bot(name, movetime=1.0)   # movetime is a placeholder: each move gets its own budget


def play(task):
    """One game on a clock. Returns a result record, or an error record if it failed."""
    try:
        return play_game(task)
    except Exception as e:
        return dict(white=task[0], black=task[1], error=f"{type(e).__name__}: {e}"[:500])
    finally:
        for engine in _engines.values():
            engine.quit()


def launch(task):
    """Play one game in its own process (Engine/tournament.py --game), which loads only
    its two players and exits when the game ends. Long-lived workers that swapped models in
    and out (freeing and recording CUDA graphs) hit "illegal memory access" after an hour,
    a sticky error that failed every later game on that worker; keeping every model ran the
    11 GB GPU out of memory; and multiprocessing.Pool(maxtasksperchild=1) stopped starting
    workers after a few dozen games, leaving their DeepMind processes behind."""
    return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--game", json.dumps(task)],
                            stdout=subprocess.PIPE, text=True)


def collect(proc, task):
    """The finished game process's result record."""
    out = proc.stdout.read()
    proc.wait()
    lines = [l for l in out.splitlines() if l.startswith(RESULT)]
    if not lines:
        return dict(white=task[0], black=task[1], error=f"game process exited with {proc.returncode} and no result")
    return json.loads(lines[-1][len(RESULT):])


def play_game(task):
    white_name, black_name, fen, clock = task
    board = chess.Board(fen)
    players = {chess.WHITE: player(white_name), chess.BLACK: player(black_name)}
    players_name = {chess.WHITE: white_name, chess.BLACK: black_name}
    game = chess.pgn.Game.from_board(board)
    node = game

    def choose(board, budget, left):
        p = players[board.turn]
        if isinstance(p, chess.engine.SimpleEngine) and players_name[board.turn].startswith("stockfish:"):
            limit = chess.engine.Limit(white_clock=left[chess.WHITE], black_clock=left[chess.BLACK],
                                       white_inc=clock[1], black_inc=clock[1])
            return p.play(board, limit, game=game).move   # Stockfish runs its own clock
        if isinstance(p, chess.engine.SimpleEngine):   # DeepMind: one pass, its time is charged all the same
            return p.play(board, chess.engine.Limit(time=budget), game=game).move
        return p.search(board, movetime=budget)[0]

    def on_move(move):
        nonlocal node
        node = node.add_variation(move)

    result, termination = play_clocked(board, choose, clock, MAX_MOVES, on_move)
    game.headers.update(White=white_name, Black=black_name, Result=result, Termination=termination,
                        TimeControl=f"{clock[0]:g}+{clock[1]:g}")
    return dict(white=white_name, black=black_name, result=result, termination=termination,
                plies=board.ply(), fen=fen, pgn=str(game), time=time.time())


def score_of(record):
    """White's score."""
    return {"1-0": 1.0, "0-1": 0.0}.get(record["result"], 0.5)


def fit(names, records, anchor):
    """MLE Elo ratings (anchor = 0) with a weak Gaussian prior, and 95% intervals from the Hessian."""
    idx = {n: i for i, n in enumerate(names)}
    games = [(idx[r["white"]], idx[r["black"]], score_of(r)) for r in records
             if r["white"] in idx and r["black"] in idx]
    free = [i for i in range(len(names)) if names[i] != anchor]
    k = math.log(10) / 400
    r = np.zeros(len(names))
    for _ in range(100):   # Newton's method on the free ratings
        grad, hess = -r / PRIOR_SD ** 2, np.diag(np.full(len(names), -1 / PRIOR_SD ** 2))
        for w, b, s in games:
            e = 1 / (1 + 10 ** ((r[b] - r[w]) / 400))
            grad[w] += k * (s - e)
            grad[b] -= k * (s - e)
            h = k * k * e * (1 - e)
            hess[w, w] -= h; hess[b, b] -= h; hess[w, b] += h; hess[b, w] += h
        g, H = grad[free], hess[np.ix_(free, free)]
        step = np.linalg.solve(H, g)
        r[free] -= step
        if np.abs(step).max() < 0.01:
            break
    cov = np.linalg.inv(-hess[np.ix_(free, free)])
    margin = np.zeros(len(names))
    margin[free] = 1.96 * np.sqrt(np.diag(cov))
    return r, margin


def standings(names, records, anchor):
    r, margin = fit(names, records, anchor)
    played = {n: [0, 0.0] for n in names}
    for rec in records:
        s = score_of(rec)
        for n, sc in ((rec["white"], s), (rec["black"], 1 - s)):
            if n in played:
                played[n][0] += 1
                played[n][1] += sc
    width = max(len(n) for n in names)
    levels = {i: int(n.split(":", 1)[1]) for i, n in enumerate(names) if n.startswith("stockfish:") and played[n][0]}
    offset = np.mean([lvl - r[i] for i, lvl in levels.items()]) if levels else None
    lines = [f"{len(records)} games, Elo relative to {anchor} (95% interval)"
             + (", and on Stockfish's UCI_Elo scale" if levels else ""), ""]
    sf = f"  {'SF':>5}" if levels else ""
    lines.append(f"{'#':>2}  {'bot':<{width}}  {'Elo':>6}  {'±':>5}{sf}  {'games':>5}  {'score':>6}")
    for rank, i in enumerate(np.argsort(-r), 1):
        n = names[i]
        g, s = played[n]
        pct = f"{100 * s / g:.0f}%" if g else "-"
        sf = f"  {r[i] + offset:>5.0f}" if levels else ""
        lines.append(f"{rank:>2}  {n:<{width}}  {r[i]:>+6.0f}  {margin[i]:>5.0f}{sf}  {g:>5}  {pct:>6}")
    # head-to-head: row's score against column, in games
    lines += ["", "head to head (row's points / games against column):"]
    order = [names[i] for i in np.argsort(-r)]
    pair = {}
    for rec in records:
        s = score_of(rec)
        for a, b, sc in ((rec["white"], rec["black"], s), (rec["black"], rec["white"], 1 - s)):
            p = pair.setdefault((a, b), [0.0, 0])
            p[0] += sc; p[1] += 1
    short = [str(j + 1) for j in range(len(order))]
    lines.append(f"{'':>{width + 4}}" + "".join(f"{c:>9}" for c in short))
    for j, a in enumerate(order):
        cells = []
        for b in order:
            p = pair.get((a, b))
            cells.append("-" if a == b else f"{p[0]:g}/{p[1]}" if p else ".")
        lines.append(f"{j + 1:>2}  {a:<{width}}" + "".join(f"{c:>9}" for c in cells))
    return "\n".join(lines)


def pick_pair(pairs, names, records, anchor, rng, pending=(), deprioritize=()):
    """A pair, with probability proportional to p(1-p) under the current ratings
    (plus a floor, so every pair keeps playing a little). Unplayed pairs go first,
    counting games in progress (pending: their records' white and black) as played."""
    counts = {p: 0 for p in pairs}
    for rec in list(records) + list(pending):
        p = tuple(sorted((rec["white"], rec["black"])))
        if p in counts:
            counts[p] += 1
    fewest = min(counts.values())
    if fewest < 2:   # every pair plays one opening (both colours) before any weighting
        return [p for p in pairs if counts[p] == fewest][rng.integers(sum(counts[p] == fewest for p in pairs))]
    r, _ = fit(names, records, anchor)
    idx = {n: i for i, n in enumerate(names)}
    w = []
    for a, b in pairs:
        e = 1 / (1 + 10 ** ((r[idx[b]] - r[idx[a]]) / 400))
        w.append(max(e * (1 - e), 0.02) / math.sqrt(1 + counts[(a, b)] / 50)
                 * (0.1 if a in deprioritize and b in deprioritize else 1))
    w = np.array(w) / sum(w)
    return pairs[rng.choice(len(pairs), p=w)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bots", nargs="*", help="bot names from Engine/bots.py, or deepmind:270M etc.")
    parser.add_argument("--clock", default="1+1", help="MINUTES+INCREMENT seconds (default 1+1)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="logs/tournament/bullet.jsonl")
    parser.add_argument("--anchor", help="bot rated 0 (default deepmind:270M if playing, else the first bot)")
    parser.add_argument("--openings", type=int, default=500, help="book openings to draw from (default 500)")
    parser.add_argument("--plies", type=int, default=10, help="opening length in plies (default 10)")
    parser.add_argument("--games", type=int, help="stop after this many games in all (default: run until stopped)")
    parser.add_argument("--report", type=int, default=10, help="print the table every this many games")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deprioritize", nargs="+", default=[], metavar="BOT",
                        help="pairs of two of these bots get a tenth of the weight")
    parser.add_argument("--standings", metavar="JSONL", help="print the table for this results file and exit")
    parser.add_argument("--game", help=argparse.SUPPRESS)   # internal: play this one game (JSON task)
    args = parser.parse_args()
    # SIGTERM (kill) unwinds like Ctrl-C, so the finally blocks stop games and their engines
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    if args.game:
        white, black, fen, clock = json.loads(args.game)
        print(RESULT + json.dumps(play((white, black, fen, tuple(clock)))), flush=True)
        return

    records = []
    path = args.standings or args.out
    if os.path.exists(path):
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
    if args.standings:
        names = args.bots or sorted({n for r in records for n in (r["white"], r["black"])})
        anchor = args.anchor or ("deepmind:270M" if "deepmind:270M" in names else names[0])
        print(standings(names, records, anchor))
        return

    names = args.bots
    if len(names) < 2:
        raise SystemExit("need at least two bots")
    anchor = args.anchor or ("deepmind:270M" if "deepmind:270M" in names else names[0])
    clock = parse_clock(args.clock)
    pairs = list(itertools.combinations(sorted(names), 2))
    openings = [b.fen() for b in book_openings(args.openings, args.plies, args.seed)]
    rng = np.random.default_rng(args.seed + len(records))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"{len(names)} bots, {len(pairs)} pairs, {args.clock} clock, {args.workers} workers, "
          f"{len(records)} games already in {args.out}", flush=True)

    # each task is a pair of games (one opening, both colours), queued as two
    queue, running = [], []   # tasks waiting; (process, task, start time) playing
    def refill():
        while not queue:
            pending = [dict(white=t[0], black=t[1]) for t in queue + [r[1] for r in running]]
            a, b = pick_pair(pairs, names, records, anchor, rng, pending, args.deprioritize)
            fen = openings[rng.integers(len(openings))]
            queue.extend([(a, b, fen, clock), (b, a, fen, clock)])

    # a game can't take longer than both clocks plus every increment, plus loading time
    game_limit = 2 * (clock[0] + MAX_MOVES * clock[1]) + 300
    done = failures = 0
    with open(args.out, "a") as out:
        try:
            while args.games is None or done < args.games:
                while len(running) < args.workers:
                    refill()
                    task = queue.pop(0)
                    running.append((launch(task), task, time.time()))
                finished = [r for r in running if r[0].poll() is not None or time.time() - r[2] > game_limit]
                if not finished:
                    time.sleep(0.5)
                    continue
                for proc, task, started in finished:
                    running.remove((proc, task, started))
                    if proc.poll() is None:
                        proc.kill()
                        rec = dict(white=task[0], black=task[1], error=f"no result after {game_limit:.0f} s: killed")
                    else:
                        rec = collect(proc, task)
                    if "error" in rec:   # not counted; logged, and the scheduler picks the pair again later
                        print(f"{rec['white']} - {rec['black']}  failed: {rec['error']}", flush=True)
                        failures += 1
                        if failures >= 20:
                            raise SystemExit("20 games in a row failed: stopping")
                        continue
                    failures = 0
                    records.append(rec)
                    out.write(json.dumps(rec) + "\n"); out.flush()
                    done += 1
                    print(f"{rec['white']} - {rec['black']}  {rec['result']}  ({rec['termination']}, "
                          f"{rec['plies']} plies)", flush=True)
                    if done % args.report == 0:
                        table = standings(names, records, anchor)
                        print("\n" + table + "\n", flush=True)
                        with open(args.out + ".standings.txt", "w") as f:
                            f.write(table + "\n")
        finally:   # on Ctrl-C, SIGTERM or an error: games in progress are dropped
            for proc, _, _ in running:
                proc.terminate()
    table = standings(names, records, anchor)
    print("\n" + table)
    with open(args.out + ".standings.txt", "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
