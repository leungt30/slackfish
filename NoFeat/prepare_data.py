"""
Encode the raw evaluation CSVs into one compact .npz for the transformer.

    uv run python -m NoFeat.prepare_data                       # all rows
    uv run python -m NoFeat.prepare_data --limit 50000 --out data/nofeat_small.npz
    uv run python -m NoFeat.prepare_data --lichess /archive/lichess/lichess_db_eval.jsonl.zst \
        --out /data/slackfish/nofeat_all          # a packed folder (see below)
    uv run python -m NoFeat.prepare_data --lichess /archive/lichess/lichess_db_eval.jsonl.zst \
        --lichess-games /archive/lichess/lichess_db_standard_rated_2026-09.pgn.zst --out /data/slackfish/nofeat_games

Lichess (--lichess): the Lichess evaluation database (database.lichess.org,
lichess_db_eval.jsonl.zst, CC0), streamed through `zstd -dc` so it is never
unpacked to disk. Each position uses its deepest evaluation's best line (scores
are from White's perspective, like the CSVs). --lichess-frac keeps a random
fraction of positions and --min-depth drops shallow evaluations. Its positions
are tagged "lichess" and are added to the CSV sources (--sources).

Lichess games (--lichess-games, may be repeated): a monthly standard games dump
(lichess_db_standard_rated_YYYY-MM.pgn.zst), also streamed. Only games with server
analysis (~6%) carry evaluations, as [%eval 0.17] / [%eval #-3] comments: the
position after that move, from White's perspective, with no depth given (Lichess
analyses the first 300 plies). Games without one are skipped by a substring search
before anything is parsed; the others are replayed with python-chess and every
annotated position that is not checkmate or stalemate is kept, from ply 15 on
(--games-min-ply): Lichess takes earlier ones from its cloud evals, the --lichess
source. --limit counts games read, with or without evals. The same position
recurs across games (openings above all), so these positions, tagged
"lichess_games", are deduplicated by a 64-bit hash of their tokens, each keeping
the mean of its evaluations. A packed folder gets the rows as they arrive and is
compacted in place at the end (memory: 12 bytes per position for the hashes and
evals, plus ~3x that while sorting them, ~20 GB for 400M positions).
--games-skip-lichess also drops those already in --lichess.

Output formats (--out):
  *.npz     the arrays below, built in memory. Fine for the CSVs (16.6M positions).
  a folder  "packed": the same arrays as raw files written as they are encoded, so
            memory stays small, with squares packed two per byte (uint8[N, 32], see
            encode.pack_squares) to halve their size: ~38 bytes per position, so all
            of Lichess (~410M positions, ~16 GB) still fits in RAM for training.
            Files: squares.bin castling.bin turn.bin eval.bin source.bin meta.json.

Every position is tagged with the file it came from, so training can pick which
sources to use (NoFeat.train --sources) without re-encoding.

Output arrays (N rows):
    squares       uint8[N, 64]   square tokens (see encode.py)
    castling      uint8[N]       castling token
    turn          uint8[N]       0 white to move, 1 black to move
    eval          float32[N]     centipawns from White's perspective;
                                 mate in n -> +-(MATE_CP - |n|)
    source        uint8[N]       index into source_names
    source_names  str[S]         e.g. ["games", "random", "tactics"]
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
from multiprocessing import Pool

import chess
import numpy as np
import pandas as pd

from NoFeat.encode import EP_TOKEN, encode_fens, pack_squares

MATE_CP = 32000
# source name -> CSV. Order matters: a source's index is stored per position.
SOURCES = {
    "games": "data/chessData.csv",       # positions from real games, Stockfish 11 depth 22
    "random": "data/random_evals.csv",   # positions after random moves
    "tactics": "data/tactic_evals.csv",  # puzzle positions (sharp, many forced mates)
}
# not CSVs: lichess is the --lichess JSONL, lichess_games the --lichess-games PGNs
SOURCE_NAMES = list(SOURCES) + ["lichess", "lichess_games"]
LICHESS_CHUNK = 50_000
GAMES_BLOCK = 1 << 23    # bytes of PGN per worker task (~3,500 games)
PGN_TOKEN = re.compile(r"\{([^}]*)\}|(\S+)")   # a comment, or a move / move number / result
PGN_EVAL = re.compile(r"\[%eval ([^\],\s]+)")


def parse_eval(s: str) -> float:
    s = str(s).strip()
    if "#" in s:
        mate = s.split("#")[1]
        n = int(mate)
        winning = not mate.startswith("-")   # "#+0" / "#-0" are already mate
        return (MATE_CP - abs(n)) * (1 if winning else -1)
    return float(s)


def mate_cp(n: int) -> float:
    """Lichess mate in n (positive: White mates) -> centipawns, as parse_eval does."""
    return (MATE_CP - abs(n)) * (1 if n > 0 else -1)


def encode_lichess_lines(args):
    """Raw JSONL lines -> (squares, castling, turn, evals) for the positions whose
    deepest evaluation reaches min_depth."""
    lines, min_depth = args
    fens, evals = [], []
    for line in lines:
        row = json.loads(line)
        best = max(row["evals"], key=lambda e: e["depth"])
        pv = best["pvs"][0]
        if best["depth"] < min_depth or pv.get("mate") == 0:
            continue
        fens.append(row["fen"])
        evals.append(pv["cp"] if "cp" in pv else mate_cp(pv["mate"]))
    return *encode_fens(fens), np.array(evals, dtype=np.float32)


def read_lichess(path, frac, min_depth, seed, limit, pool):
    """Stream the Lichess .jsonl.zst and yield (squares, castling, turn, evals) parts
    for a random frac of its positions."""
    rng = np.random.default_rng(seed)
    zstd = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, bufsize=1 << 24)

    def chunks():
        read = kept = 0
        batch = []
        for line in zstd.stdout:
            read += 1
            if frac >= 1 or rng.random() < frac:
                batch.append(line)
                kept += 1
            if len(batch) == LICHESS_CHUNK:
                yield batch, min_depth
                batch = []
            if read % 10_000_000 == 0:
                print(f"  lichess: read {read:,} lines, sampled {kept:,}", flush=True)
            if limit and kept >= limit:
                break
        if batch:
            yield batch, min_depth

    yield from pool.imap(encode_lichess_lines, chunks())
    zstd.kill()


def board_bits(board):
    """A chess.Board as the ints encode_boards needs: what board.fen(en_passant="legal")
    holds, without building the string (half the cost of encode_games)."""
    rights = board.clean_castling_rights()
    return (board.occupied_co[chess.BLACK], board.pawns, board.knights, board.bishops, board.rooks,
            board.queens, board.kings,
            64 if board.ep_square is None or not board.has_legal_en_passant() else board.ep_square,
            bool(rights & chess.BB_H1) | bool(rights & chess.BB_A1) << 1
            | bool(rights & chess.BB_H8) << 2 | bool(rights & chess.BB_A8) << 3,
            int(board.turn == chess.BLACK))


def encode_boards(bits):
    """board_bits rows -> (squares, castling, turn), exactly as encode_fens would give."""
    rows = np.array(bits, dtype=np.uint64).reshape(-1, 10)
    planes = np.unpackbits(np.ascontiguousarray(rows[:, :7]).view(np.uint8).reshape(-1, 7, 8),
                           axis=2, bitorder="little")   # [N, 7, 64], bit i = square i
    squares = (planes[:, 1:] * np.arange(1, 7, dtype=np.uint8)[:, None]).sum(1, dtype=np.uint8)
    squares += 6 * (planes[:, 0] & (squares > 0))
    ep = rows[:, 7] < 64
    squares[ep, rows[ep, 7].astype(np.int64)] = EP_TOKEN
    return squares, rows[:, 8].astype(np.uint8), rows[:, 9].astype(np.uint8)


def analysed_games(text):
    """The games in PGN text (bytes, whole games) that have [%eval]s, found by substring
    search alone: most games have none, and are never parsed."""
    pos = text.find(b"[%eval")
    while pos >= 0:
        end = text.find(b"\n[Event ", pos) + 1 or len(text)
        yield text[text.rfind(b"\n[Event ", 0, pos) + 1:end]
        pos = text.find(b"[%eval", end)


def encode_games(args):
    """Raw PGN text (bytes, whole games) -> (squares, castling, turn, evals) for the
    positions after its [%eval]-annotated moves, from ply min_ply on, plus the number
    of games and of analysed games."""
    text, min_ply = args
    boards, evals, analysed = [], [], 0
    for game in analysed_games(text):
        board, ply = chess.Board(), 0
        analysed += 1
        for comment, token in PGN_TOKEN.findall(game.split(b"\n\n", 1)[1].decode()):
            if token:
                if token[0].isdigit() or token == "*":   # move number or result
                    continue
                try:
                    board.push_san(token.rstrip("?!"))
                except ValueError:
                    break
                ply += 1
                continue
            score = PGN_EVAL.search(comment)
            if score is None or ply < min_ply:
                continue
            score = score.group(1)
            # as encode_lichess_lines drops mate == 0: no checkmate or stalemate positions
            if score in ("#0", "#-0") or not any(board.generate_legal_moves()):
                continue
            boards.append(board_bits(board))
            evals.append(mate_cp(int(score[1:])) if score[0] == "#" else round(float(score) * 100))
    games = text.count(b"\n[Event ") + text.startswith(b"[Event ")
    return *encode_boards(boards), np.array(evals, dtype=np.float32), games, analysed


def read_lichess_games(path, min_ply, limit, pool, workers, stats):
    """Stream a Lichess games .pgn.zst and yield (squares, castling, turn, evals) parts
    for its analysed games, counting games read and analysed in stats."""
    zstd = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, bufsize=1 << 24)
    # Pool.imap reads its input as fast as it can, so cap the blocks in flight
    slots = threading.Semaphore(4 * workers)

    def chunks():
        tail, read = b"", 0
        while True:
            block = zstd.stdout.read(GAMES_BLOCK)
            text = tail + block
            if block:   # keep the last game, maybe cut off, for the next block
                cut = text.rfind(b"\n[Event ") + 1
                text, tail = text[:cut], text[cut:]
            if limit:
                games = text.count(b"\n[Event ") + bool(text)   # text starts at a game
                if read + games >= limit:   # cut it after the limit-th game
                    cut = 0
                    for _ in range(limit - read):
                        cut = text.find(b"\n[Event ", cut) + 1 or len(text)
                    text, block = text[:cut], b""
                read += games
            slots.acquire()
            yield text, min_ply
            if not block:
                break

    for squares, castling, turn, evals, games, analysed in pool.imap(encode_games, chunks()):
        slots.release()
        if (stats["games"] + games) // 10_000_000 > stats["games"] // 10_000_000:
            print(f"  lichess_games: read {stats['games'] + games:,} games, "
                  f"{stats['analysed'] + analysed:,} analysed", flush=True)
        stats["games"] += games
        stats["analysed"] += analysed
        yield squares, castling, turn, evals
    zstd.kill()


def position_hashes(squares, castling, turn):
    """64-bit hash of each position's tokens (squares, castling, turn), for deduplication."""
    def mix(x):   # the splitmix64 finalizer
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))

    words = np.ascontiguousarray(squares).view(np.uint64)   # [N, 8]: 8 squares per word
    h = mix(castling.astype(np.uint64) | (turn.astype(np.uint64) << np.uint64(8)))
    for i in range(8):
        h = mix(h ^ words[:, i])
    return h


def dedup(hashes, evals, exclude=None):
    """(rows, evals) keeping the first row of each distinct hash, in order, labelled
    with the mean eval of all its rows; hashes found in exclude are dropped."""
    if not len(hashes):
        return np.zeros(0, np.int64), evals
    order = np.argsort(hashes, kind="stable")   # stable: equal hashes keep their first row first
    hashes = hashes[order]
    first = np.concatenate(([True], hashes[1:] != hashes[:-1]))
    group = np.cumsum(first) - 1
    mean = np.bincount(group, weights=evals[order]) / np.bincount(group)
    rows = order[first]
    if exclude is not None:
        new = ~np.isin(hashes[first], exclude)
        rows, mean = rows[new], mean[new]
    by_row = np.argsort(rows)
    return rows[by_row], mean[by_row].astype(np.float32)


class NpzSink:
    """Collects parts in memory and saves one .npz."""

    def __init__(self, path):
        self.path, self.parts, self.n = path, [], 0

    def add(self, squares, castling, turn, evals, source):
        self.parts.append((squares, castling, turn, evals, source))
        self.n += len(evals)

    def keep_rows(self, start, rows, evals):
        """Of the rows from start on, keep only rows (offsets from start), relabelled evals."""
        arrays = [np.concatenate([p[i] for p in self.parts]) for i in range(5)]
        keep = np.concatenate((np.arange(start), start + rows))
        arrays = [a[keep] for a in arrays]
        arrays[3][start:] = evals
        self.parts, self.n = [tuple(arrays)], len(keep)

    def close(self, names):
        squares, castling, turn, evals, source = (np.concatenate([p[i] for p in self.parts]) for i in range(5))
        np.savez(self.path, squares=squares, castling=castling, turn=turn, eval=evals,
                 source=source, source_names=np.array(names))
        return evals


class PackedSink:
    """Appends each part to raw files in a folder as it arrives (see the docstring)."""
    FILES = ("squares", "castling", "turn", "eval", "source")

    def __init__(self, path):
        self.path, self.n = path, 0
        os.makedirs(path, exist_ok=True)
        self.files = {name: open(os.path.join(path, f"{name}.bin"), "wb") for name in self.FILES}
        self.eval_sample = []   # (first row, every 100th eval) per part, for the summary

    def add(self, squares, castling, turn, evals, source):
        arrays = dict(squares=pack_squares(squares), castling=castling, turn=turn,
                      eval=evals.astype(np.float32), source=source)
        for name, array in arrays.items():
            self.files[name].write(np.ascontiguousarray(array).tobytes())
        self.eval_sample.append((self.n, evals[::100]))
        self.n += len(evals)

    def keep_rows(self, start, rows, evals):
        """Of the rows from start on, keep only rows (offsets from start), relabelled
        evals, compacting the files in place a slice at a time (rows is sorted, so a
        row is always read before it is overwritten)."""
        for name, f in self.files.items():
            f.flush()
            width = 32 if name == "squares" else 1
            dtype = np.float32 if name == "eval" else np.uint8
            data = np.memmap(f.name, dtype=dtype, mode="r+").reshape(-1, width)
            for i in range(0, len(rows), 10_000_000):
                part = rows[i:i + 10_000_000]
                data[start + i:start + i + len(part)] = (evals[i:i + len(part)].reshape(-1, 1) if name == "eval"
                                                         else data[start + part])
            data.flush()
            del data
            f.seek(0, os.SEEK_SET)
            f.truncate((start + len(rows)) * width * np.dtype(dtype).itemsize)
            f.seek(0, os.SEEK_END)
        self.eval_sample = [(at, e) for at, e in self.eval_sample if at < start] + [(start, evals[::100])]
        self.n = start + len(rows)

    def close(self, names):
        for f in self.files.values():
            f.close()
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump({"format": "packed-v1", "n": self.n, "source_names": names}, f)
        return np.concatenate([e for _, e in self.eval_sample])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES),
                        help="which CSVs to encode (default: all; choose at training time instead)")
    parser.add_argument("--limit", type=int, help="max rows per file (for --lichess-games: games read)")
    parser.add_argument("--out", default="data/nofeat.npz")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--lichess", help="also encode this lichess_db_eval.jsonl.zst")
    parser.add_argument("--lichess-frac", type=float, default=1.0,
                        help="keep this random fraction of Lichess positions (default: all)")
    parser.add_argument("--min-depth", type=int, default=20,
                        help="drop Lichess positions evaluated below this depth (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="for the Lichess sample")
    parser.add_argument("--lichess-games", action="append", default=[], metavar="PGN_ZST",
                        help="also encode the analysed positions of this lichess_db_standard_rated_*.pgn.zst "
                             "(may be repeated)")
    parser.add_argument("--games-min-ply", type=int, default=15,
                        help="drop --lichess-games positions before this ply (default 15: Lichess fills "
                             "those from its cloud evals, i.e. --lichess; 0 keeps all)")
    parser.add_argument("--games-skip-lichess", action="store_true",
                        help="drop --lichess-games positions that are also in --lichess")
    args = parser.parse_args()
    for path in [args.lichess] + args.lichess_games:
        if path and not os.path.exists(path):
            raise SystemExit(f"{path} not found")
    if args.games_skip_lichess and not (args.lichess and args.lichess_games):
        raise SystemExit("--games-skip-lichess needs --lichess and --lichess-games")

    names = SOURCE_NAMES
    dfs = []
    for name in args.sources:
        path = SOURCES[name]
        if not os.path.exists(path):
            print(f"{path} not found, skipping '{name}'")
            continue
        df = pd.read_csv(path, nrows=args.limit, usecols=["FEN", "Evaluation"])
        df["source"] = names.index(name)
        print(f"{name:8s} {path}: {len(df):,} rows")
        dfs.append(df)
    df = (pd.concat(dfs, ignore_index=True) if dfs
          else pd.DataFrame({"FEN": [], "Evaluation": [], "source": []}))
    fens = df["FEN"].tolist()

    os.makedirs(os.path.dirname(args.out.rstrip("/")) or ".", exist_ok=True)
    sink = NpzSink(args.out) if args.out.endswith(".npz") else PackedSink(args.out)
    start = time.time()
    chunks = [fens[i:i + 50_000] for i in range(0, len(fens), 50_000)]
    with Pool(args.workers) as pool:
        parts = pool.map(encode_fens, chunks) or [encode_fens([])]
        sink.add(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                 np.concatenate([p[2] for p in parts]),
                 df["Evaluation"].map(parse_eval).to_numpy(np.float32), df["source"].to_numpy(np.uint8))
        print(f"Encoded {len(fens):,} positions in {time.time() - start:.0f}s")
        del parts, df, fens

        lichess_hashes = [np.zeros(0, np.uint64)]   # for --games-skip-lichess
        if args.lichess:
            start, count = time.time(), 0
            lichess = names.index("lichess")
            for squares, castling, turn, evals in read_lichess(args.lichess, args.lichess_frac,
                                                               args.min_depth, args.seed, args.limit, pool):
                sink.add(squares, castling, turn, evals, np.full(len(evals), lichess, np.uint8))
                if args.games_skip_lichess:
                    lichess_hashes.append(position_hashes(squares, castling, turn))
                count += len(evals)
            print(f"lichess  {args.lichess}: {count:,} positions (depth >= {args.min_depth}) "
                  f"in {time.time() - start:.0f}s")

        if args.lichess_games:
            # written as they arrive, then deduplicated across all the PGNs (see the docstring)
            games, first = names.index("lichess_games"), sink.n
            hashes, game_evals = [np.zeros(0, np.uint64)], [np.zeros(0, np.float32)]
            for path in args.lichess_games:
                start, count, stats = time.time(), 0, {"games": 0, "analysed": 0}
                for squares, castling, turn, evals in read_lichess_games(path, args.games_min_ply, args.limit,
                                                                         pool, args.workers, stats):
                    sink.add(squares, castling, turn, evals, np.full(len(evals), games, np.uint8))
                    hashes.append(position_hashes(squares, castling, turn))
                    game_evals.append(evals)
                    count += len(evals)
                print(f"lichess_games {path}: {stats['games']:,} games, {stats['analysed']:,} analysed "
                      f"({stats['analysed'] / max(stats['games'], 1):.1%}), {count:,} positions "
                      f"in {time.time() - start:.0f}s")
            start, total = time.time(), sink.n - first
            exclude = np.unique(np.concatenate(lichess_hashes)) if args.games_skip_lichess else None
            del lichess_hashes
            rows, evals = dedup(np.concatenate(hashes), np.concatenate(game_evals), exclude)
            del hashes, game_evals, exclude
            sink.keep_rows(first, rows, evals)
            print(f"lichess_games: kept {len(rows):,} distinct positions of {total:,} "
                  f"({1 - len(rows) / max(total, 1):.1%} duplicates"
                  f"{', or already in --lichess' if args.games_skip_lichess else ''}) "
                  f"in {time.time() - start:.0f}s")

    evals = sink.close(names)
    mates = np.abs(evals) > MATE_CP - 1000
    sampled = "" if isinstance(sink, NpzSink) else " (stats from a 1% sample)"
    print(f"Saved {args.out}: {sink.n:,} positions, {mates.mean():.1%} forced mates, "
          f"median |eval| {np.median(np.abs(evals[~mates])):.0f} cp{sampled}")


if __name__ == "__main__":
    main()
