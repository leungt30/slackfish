"""Stockfish preparation against an opponent's observed openings (run with nice, 2 threads).

    curl -H "Authorization: Bearer $(cat .lichess_token)" -H "Accept: application/x-ndjson" \\
         "https://lichess.org/api/games/user/NAME?max=400&moves=true&rated=true" -o NAME.ndjson
    nice -n 19 python Engine/prep_opponent.py NAME NAME.ndjson NAME-lines.jsonl [d2d4 ...]
    python Engine/make_book.py NAME-lines.jsonl Models/Books/NAME.bin

Walks the tree of positions for each colour the opponent played. Extra arguments are UCI moves
played from the start (e.g. d2d4): only that opening is walked, with the opponent's colour implied.
For each colour we play: walk the tree of positions. Their moves branch on what they actually played
(>= MIN_N games and >= 8% of that position). Our moves: Cerebellum's top move if it has the position,
else Stockfish's best (MultiPV 2, SECS each). Writes bonga_prep.jsonl (one analysed position per line)."""
import collections, json, shutil, sys, time
import chess, chess.engine, chess.polyglot
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPPONENT, GAMES, OUT, *PREFIX = sys.argv[1:]
SECS, MAX_PLY, MIN_N, MAX_NODES = (int(os.environ.get(k, d)) for k, d in (("SECS", 30), ("MAX_PLY", 16), ("MIN_N", 4), ("MAX_NODES", 120)))
SHARE = float(os.environ.get("SHARE", 0.08))   # an opponent move is followed if played in at least this share of the games
games = [json.loads(l) for l in open(GAMES)]
nm = lambda g, c: g["players"][c].get("user", {}).get("name")
theirs = {"white": collections.defaultdict(collections.Counter), "black": collections.defaultdict(collections.Counter)}
for g in games:
    c = "white" if nm(g, "white") == OPPONENT else "black"
    b = chess.Board()
    for san in g["moves"].split()[:MAX_PLY]:
        m = b.parse_san(san)
        if (b.turn == chess.WHITE) == (c == "white"):
            theirs[c][b.epd()][m.uci()] += 1
        b.push(m)
book = chess.polyglot.open_reader(f"{ROOT}/Models/Books/Cerebellum3Merge.bin")
eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
eng.configure({"Threads": 2, "Hash": 256})
out = open(OUT, "a")
done, n_an, t0 = {}, 0, time.time()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def walk(b, c):   # c = the opponent's colour
    global n_an
    key = (b.epd(), c)
    if b.ply() >= MAX_PLY or b.is_game_over() or key in done: return
    done[key] = 1
    if (b.turn == chess.WHITE) == (c == "white"):   # their move: branch on what they play
        cnt = theirs[c].get(b.epd())
        if not cnt: return
        tot = sum(cnt.values())
        for u, n in cnt.most_common():
            if n >= MIN_N and n >= SHARE * tot:
                b.push_uci(u); walk(b, c); b.pop()
        return
    es = list(book.find_all(b))
    if es:
        mv = max(es, key=lambda e: e.weight).move; src = "cerebellum"
    else:
        if n_an >= MAX_NODES: return
        info = eng.analyse(b, chess.engine.Limit(time=SECS), multipv=2)
        mv = info[0]["pv"][0]; src = "stockfish"; n_an += 1
        rec = {"fen": b.fen(), "us": "black" if c == "white" else "white", "best": b.san(mv), "uci": mv.uci(),
               "cp_white": info[0]["score"].white().score(mate_score=10000), "depth": info[0].get("depth"),
               "alt": [{"san": b.san(i["pv"][0]), "cp_white": i["score"].white().score(mate_score=10000)} for i in info[1:]],
               "moves": [m.uci() for m in b.move_stack]}
        out.write(json.dumps(rec) + "\n"); out.flush()
        log(f"#{n_an} {len(b.move_stack)}p {rec['us']} plays {rec['best']} ({rec['cp_white']:+d}cp white, d{rec['depth']})")
    b.push(mv); walk(b, c); b.pop()
if PREFIX:
    start = chess.Board()
    for u in PREFIX:
        start.push_uci(u)
    c = "black" if len(PREFIX) % 2 else "white"   # the opponent moves next
    log("tree:", OPPONENT, "plays", c, "after", " ".join(PREFIX))
    walk(start, c)
else:
    for c in ("white", "black"):
        log("tree:", OPPONENT, "plays", c); walk(chess.Board(), c)
eng.quit()
log(f"done: {n_an} positions analysed in {(time.time()-t0)/60:.0f} min")
