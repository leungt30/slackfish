"""
Hard-position mining: run Stockfish over the bot's Lichess games (logs/games/*.pgn,
saved by Engine/lichess.py) and list the moves it got wrong, with what the bot
thought of them, as candidate training positions.

    uv run python Engine/mine_hard.py logs/games/*.pgn --depth 18 --threshold 150 --out data/hard_positions.csv
    uv run python Engine/mine_hard.py logs/games/*.pgn --max-games 20      # a batch; run again for the next
    uv run python Engine/mine_hard.py --summary --out data/hard_positions.csv   # the summary only, no Stockfish

For each of our moves, Stockfish (multipv 1, to --depth) evaluates the position
before it, which gives the best move and its value, and, unless we played that move,
the position after it. A move is flagged when it loses more than --threshold
centipawns against the best move, from our side (drop_cp; drop_wp is the same in
win probability, small for a move that only deepens a lost or won position). Book and tablebase moves are
skipped (--include-book analyses book moves too).

One CSV row per flagged move: the position before it (FEN), our move and Stockfish's,
Stockfish's eval before and after (centipawns and win probability, White's point of
view), the bot's own score from the PGN comment (White's point of view, the value its
search gave the move it chose), its depth, the seconds it used and the clock left, the
opponent and the result. Win probability is sigmoid(cp / 400), the mapping of the
training target (NoFeat/train.py --target winprob, --wp-scale 400); forced mates
count as 10000 cp less the moves to mate.

The summary sorts the flagged moves by
  phase      opening (28+ men on the board), endgame (14 or fewer) or middlegame;
  our view   the bot's score against Stockfish's eval of the position its move led to,
             both from our side:
               overrated  it scored the move more than --agree cp above Stockfish: it
                          misjudged where the move leads (the network's evaluation, or
                          a refutation beyond its depth): the evaluator's errors
               knew       within --agree cp: it knew what the move was worth, but missed
                          that the best move was better (that line underrated or not
                          searched deep enough)
               underrated more than --agree cp below Stockfish
               no score   no comment (a game resumed after a restart, or an older PGN)
  clock      short when the clock after the move was under --short of the starting
             clock (default 15%).

Resumable: games already analysed (listed in OUT.done, or with rows in OUT) are
skipped, so the same command picks up where it stopped. Stockfish runs with 1 thread
and 256 MB hash under nice by default; one game takes about a minute at depth 18.
Stockfish needs to be on PATH (or pass --stockfish).
"""
import argparse
import csv
import math
import os
import re
import shutil
from collections import Counter

import chess
import chess.engine
import chess.pgn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WP_SCALE = 400   # win probability = sigmoid(cp / WP_SCALE), as NoFeat/train.py --target winprob (--wp-scale)
MATE = 10000     # centipawns for a forced mate, less the moves to it
OPENING_MEN, ENDGAME_MEN = 28, 14   # pieces and pawns on the board, both sides and kings included
# The bot's PGN comments (GamePlayer.play in lichess.py), e.g.
# "score +35, depth 6, ponder hit #2/3 at d4, 3.10s of 12.50s", "score book, ...", "score tablebase win, ..."
SCORE = re.compile(r"\bscore ([+-]?\d+|book|tablebase)")
DEPTH = re.compile(r"\bdepth (\d+)")
TIME = re.compile(r"([\d.]+)s of ([\d.]+)s")
PONDER = re.compile(r"\bponder (hit|miss)")
COLUMNS = ["game_id", "ply", "fen", "our_move", "sf_best", "sf_before_cp", "sf_after_cp", "sf_before_wp",
           "sf_after_wp", "drop_cp", "drop_wp", "our_score_cp", "our_view", "our_depth", "time_used", "budget",
           "clock_after", "clock_short", "ponder", "phase", "men", "opponent", "result", "our_result"]


def win_prob(cp):
    return 1 / (1 + math.exp(-cp / WP_SCALE))


def phase_of(men):
    return "opening" if men >= OPENING_MEN else "endgame" if men <= ENDGAME_MEN else "middlegame"


def game_id_of(game, path):
    site = game.headers.get("Site", "")
    if "lichess.org/" in site:
        return site.rstrip("/").rsplit("/", 1)[1][:8]
    return os.path.splitext(os.path.basename(path))[0]


def our_color(game, me):
    """The bot's colour: the side named me, else the side with the bot's comments, else None."""
    for color, side in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        if game.headers.get(side, "").lower() == me.lower():
            return color
    for node in game.mainline():
        if SCORE.search(node.comment):
            return not node.board().turn   # the side that just moved
    return None


def evaluate(engine, board, limit, game_id):
    """Stockfish's (White-perspective cp, best move or None) for board."""
    info = engine.analyse(board, limit, game=game_id)
    pv = info.get("pv") or [None]
    return info["score"].white().score(mate_score=MATE), pv[0]


def mine_game(engine, game, game_id, args):
    """(rows for the flagged moves, number of our moves analysed)."""
    color = our_color(game, args.me)
    if color is None:
        print(f"{game_id}: neither side is {args.me} and no bot comments, skipped")
        return [], 0
    sign = 1 if color == chess.WHITE else -1
    h = game.headers
    opponent = h.get("Black" if color == chess.WHITE else "White", "?")
    result = h.get("Result", "*")
    our_result = {"1-0": "win" if color == chess.WHITE else "loss", "0-1": "loss" if color == chess.WHITE else "win",
                  "1/2-1/2": "draw"}.get(result, "unfinished")
    try:
        base = float(h.get("TimeControl", "-").split("+")[0])
    except ValueError:
        base = None
    limit = chess.engine.Limit(depth=args.depth)
    rows, analysed = [], 0
    board = game.board()
    for node in game.mainline():
        move, ply = node.move, len(board.move_stack)
        if board.turn != color:
            board.push(move)
            continue
        note = node.comment
        score = SCORE.search(note)
        ours = score.group(1) if score else None
        if ours == "tablebase" or (ours == "book" and not args.include_book):
            board.push(move)
            continue
        analysed += 1
        before, best = evaluate(engine, board, limit, game_id)
        fen = board.fen()
        men = len(board.piece_map())
        board.push(move)
        after = before if move == best else evaluate(engine, board, limit, game_id)[0]
        drop = sign * (before - after)
        if drop <= args.threshold:
            continue
        our_cp = int(ours) if ours not in (None, "book") else None
        if our_cp is None:
            view = "book" if ours == "book" else "no score"
        else:
            error = sign * (our_cp - after)
            view = "overrated" if error > args.agree else "underrated" if error < -args.agree else "knew"
        clock = node.clock()
        depth, took, ponder = DEPTH.search(note), TIME.search(note), PONDER.search(note)
        rows.append({
            "game_id": game_id, "ply": ply, "fen": fen, "our_move": move.uci(),
            "sf_best": best.uci() if best else "", "sf_before_cp": before, "sf_after_cp": after,
            "sf_before_wp": f"{win_prob(before):.4f}", "sf_after_wp": f"{win_prob(after):.4f}", "drop_cp": drop,
            "drop_wp": f"{sign * (win_prob(before) - win_prob(after)):.4f}",
            "our_score_cp": "" if our_cp is None else our_cp, "our_view": view,
            "our_depth": depth.group(1) if depth else "", "time_used": took.group(1) if took else "",
            "budget": took.group(2) if took else "", "clock_after": "" if clock is None else f"{clock:.1f}",
            "clock_short": "" if clock is None or not base else "yes" if clock < args.short * base else "no",
            "ponder": ponder.group(1) if ponder else "", "phase": phase_of(men), "men": men,
            "opponent": opponent, "result": result, "our_result": our_result,
        })
    return rows, analysed


def read_done(out):
    """Game ids already analysed -> our moves analysed (None if only known from the CSV)."""
    done = {}
    if os.path.exists(out):
        with open(out, newline="") as f:
            for row in csv.DictReader(f):
                done.setdefault(row["game_id"], None)
    if os.path.exists(out + ".done"):
        for line in open(out + ".done"):
            if line.strip():
                game_id, _, n = line.strip().partition("\t")
                done[game_id] = int(n) if n.isdigit() else None
    return done


def summary(out):
    done = read_done(out)
    rows = list(csv.DictReader(open(out, newline=""))) if os.path.exists(out) else []
    analysed = sum(n or 0 for n in done.values())
    print(f"\n{len(rows)} flagged moves in {len(done)} games ({analysed} of our moves analysed): {out}")
    if not rows:
        return
    for title, key, order in (("phase", "phase", ["opening", "middlegame", "endgame"]),
                              ("our view", "our_view", ["overrated", "knew", "underrated", "no score", "book"]),
                              ("clock", "clock_short", ["no", "yes", ""])):
        counts = Counter(r[key] for r in rows)
        labels = {"no": "normal", "yes": "short", "": "unknown"} if key == "clock_short" else {}
        print(f"  by {title}: " + ", ".join(f"{labels.get(k, k)} {counts[k]}" for k in order if counts[k]))
    drops = sorted(int(r["drop_cp"]) for r in rows)
    print(f"  median drop {drops[len(drops) // 2]} cp; results: "
          + ", ".join(f"{k} {v}" for k, v in Counter(r["our_result"] for r in rows).most_common()))


def main():
    parser = argparse.ArgumentParser(description="Find the bot's mistakes in its games with Stockfish.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("pgns", nargs="*", help="PGN files (one or more games each), e.g. logs/games/*.pgn")
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "hard_positions.csv"))
    parser.add_argument("--stockfish", default=shutil.which("stockfish") or "stockfish")
    parser.add_argument("--depth", type=int, default=18, help="Stockfish depth per position (default 18)")
    parser.add_argument("--threshold", type=float, default=150,
                        help="flag moves losing more than this many centipawns to the best move (default 150)")
    parser.add_argument("--agree", type=float, default=100,
                        help="the bot's score 'knew' the move's value when within this many cp of Stockfish (default 100)")
    parser.add_argument("--short", type=float, default=0.15,
                        help="the clock was short under this share of the starting clock (default 0.15)")
    parser.add_argument("--me", default="slackfish-bot", help="the bot's Lichess name (default slackfish-bot)")
    parser.add_argument("--include-book", action="store_true", help="analyse book moves too")
    parser.add_argument("--max-games", type=int, help="analyse at most this many new games, then stop")
    parser.add_argument("--threads", type=int, default=1, help="Stockfish threads (default 1)")
    parser.add_argument("--hash", type=int, default=256, help="Stockfish hash in MB (default 256)")
    parser.add_argument("--summary", action="store_true", help="print the summary of --out and exit")
    args = parser.parse_args()
    if args.summary or not args.pgns:
        summary(args.out)
        return

    done = read_done(args.out)
    games = []   # (game id, game) not analysed yet
    for path in args.pgns:
        with open(path) as f:
            while (game := chess.pgn.read_game(f)) is not None:
                game_id = game_id_of(game, path)
                if game_id not in done and game.next() is not None:
                    games.append((game_id, game))
    if args.max_games is not None:
        games = games[:args.max_games]
    print(f"{len(games)} games to analyse ({len(done)} done before) at depth {args.depth}")
    if not games:
        summary(args.out)
        return
    command = [args.stockfish]
    if shutil.which("nice"):
        command = ["nice", "-n", "10"] + command   # the machine has other work: run at low priority
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(command)
    try:
        engine.configure({"Threads": args.threads, "Hash": args.hash})
        new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
        with open(args.out, "a", newline="") as out, open(args.out + ".done", "a") as marks:
            writer = csv.DictWriter(out, COLUMNS)
            if new:
                writer.writeheader()
            for i, (game_id, game) in enumerate(games, 1):
                rows, analysed = mine_game(engine, game, game_id, args)
                writer.writerows(rows)
                out.flush()
                marks.write(f"{game_id}\t{analysed}\n")   # after the rows: a crash in between leaves rows, not a gap
                marks.flush()
                print(f"[{i}/{len(games)}] {game_id}: {len(rows)} of {analysed} moves flagged"
                      f" ({game.headers.get('White')} - {game.headers.get('Black')} {game.headers.get('Result')})")
    finally:
        engine.quit()
    summary(args.out)


if __name__ == "__main__":
    main()
