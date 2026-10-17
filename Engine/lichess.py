"""
Run a SlackFish bot on a Lichess BOT account: friends challenge it from its
profile, and it can play Lichess's own Stockfish (the "AI", levels 1-8).

To run it day to day, use scripts/lichess-bot.sh (start / series / stop / status /
log; its header has notes for operating it, including for an AI assistant).

    uv run python Engine/lichess.py upgrade                  # once: turn the account into a BOT
    uv run python Engine/lichess.py play                     # accept challenges until Ctrl+C
    uv run python Engine/lichess.py play --stockfish 3 --games 4 --clock 3+2   # 4 games vs AI level 3
    uv run python Engine/lichess.py play --challenge maia1,GarboBot --games 4 --rated   # vs other bots
    uv run python Engine/lichess.py play --bot t2-w-6.41m:2 --casual-only --no-bots

Setup: make a NEW Lichess account (an account can only become a BOT before it
has played any game, and it can't go back). Logged in as it, create a token at
https://lichess.org/account/oauth/token/create with "Play games with the bot API"
(bot:play), and put it in LICHESS_TOKEN or in a .lichess_token file at the repo
root (gitignored). Then run `upgrade` once.

Games: standard chess with a clock only, at most --max-games at a time (default
1, since searches share the GPU); other challenges are declined with a reason
Lichess shows the challenger, as are clocks without increment (1+0, 5+0), where the
bot can flag (--allow-no-increment accepts them). The bot manages its Lichess clock like the GUI's
bot clock (see Bot in bots.py), but more freely: about 1/12 of its time plus the
increment per move, 1/20 under 3 minutes (--moves-left), searching by time with
no depth limit. On the opponent's
time it ponders their 3 likeliest replies, and if one is played after it was
searched to the bot's depth (the :DEPTH in --bot), it moves instantly. With 7 pieces or
fewer it plays the perfect move from Lichess's endgame tablebase instead of
searching (unless --no-tablebase or the lookup fails). It accepts a draw offer
when its last evaluation is under +0.10 for itself (a tablebase draw counts as
0), and declines otherwise (and takebacks always).

--stockfish LEVEL[,LEVEL...] and --challenge USER[,USER...] have the bot challenge
the Lichess AI and other players (usually bots) whenever it's idle, taking turns
through the list, until --games of those games are done. Games against the AI
are never rated; --rated makes the others rated. Opponents that decline twice in
a row, or don't answer within a minute twice, are dropped. Friends' challenges
are still accepted between games. When the list is done it prints the score per
opponent and stays online for challenges.
"""
import argparse
import json
import os
import threading
import time
from collections import defaultdict

import chess
import requests

from bots import BOTS, PONDER_REPLIES, ROOT, OpeningBook, SearchAborted, load_bot

API = "https://lichess.org"
TOKEN_FILE = os.path.join(ROOT, ".lichess_token")
LAG = 0.3   # seconds kept back from each move's budget for the network round trip
MOVES_LEFT = 12   # --moves-left for 3+ minute clocks (blitz games ended with 160-245 s unused at 25)
MOVES_LEFT_BULLET = 12   # ... and under 3 minutes (1+1 games ended with ~44 s of 60 unused at 20)
DRAW_BELOW = 10   # accept a draw offer when our last eval is under this many centipawns (+0.10) for us
TABLEBASE = "https://tablebase.lichess.ovh/standard"   # Lichess's free endgame tablebase (up to 7 pieces)
TABLEBASE_PIECES = 7
# Control files (scripts/lichess-bot.sh pause / resume / stop): checked twice a second
PAUSE_FILE = os.path.join(ROOT, "logs", "lichess.pause")   # finish games in progress, then start none
STOP_FILE = os.path.join(ROOT, "logs", "lichess.stop")     # ... and exit once none is in progress


def read_token():
    token = os.environ.get("LICHESS_TOKEN")
    if not token and os.path.exists(TOKEN_FILE):
        token = open(TOKEN_FILE).read().strip()
    if not token:
        raise SystemExit(f"No Lichess token: set LICHESS_TOKEN or write it to {TOKEN_FILE} (see --help)")
    return token


class Lichess:
    """The few Lichess API calls a bot needs."""

    def __init__(self, token):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def get(self, path):
        r = self.session.get(API + path, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_ndjson(self, path, params=None):
        r = self.session.get(API + path, params=params, headers={"Accept": "application/x-ndjson"}, timeout=30)
        r.raise_for_status()
        return [json.loads(line) for line in r.text.splitlines() if line.strip()]

    def post(self, path, data=None):
        """POST, retrying dropped connections and server errors (Lichess closes idle
        keep-alive connections, so the first request after a pause can fail; an
        unsent move would otherwise lose on time) and waiting out rate limits."""
        for attempt in range(6):
            try:
                r = self.session.post(API + path, data=data, timeout=15)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 5:
                    raise
                print(f"POST {path} failed ({e.__class__.__name__}), retrying")
                time.sleep(0.25 * 2 ** attempt)
                continue
            if r.status_code == 429:
                print("rate limited by Lichess, waiting a minute")
                time.sleep(60)
            elif r.status_code >= 500 and attempt < 5:
                time.sleep(0.25 * 2 ** attempt)
            else:
                break
        if not r.ok:
            raise requests.HTTPError(f"{r.status_code} {path}: {r.text.strip()}", response=r)
        return r.json() if r.content else None

    def stream(self, path):
        """Yield each event of an ndjson stream, or None for Lichess's blank
        keep-alive lines (every few seconds), so a reader can check for stopping."""
        with self.session.get(API + path, stream=True, timeout=(30, 90)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                yield json.loads(line) if line else None


def tablebase_move(board, timeout):
    """With TABLEBASE_PIECES or fewer pieces, the perfect move from Lichess's
    endgame tablebase: (move, White-perspective score, log note), or None if the
    position is too big, the lookup fails or times out, or the answer is unknown.

    The tablebase plays won endings by the shortest safe route to a capture or
    pawn move (DTZ), so it converts wins the search can't (king and queen against
    king just looks "winning" from every square to the network, so it shuffles
    checks into a repetition) and holds draws. It's a different host from the
    Bot API, so no token is sent."""
    if len(board.piece_map()) > TABLEBASE_PIECES or board.castling_rights:
        return None
    try:
        r = requests.get(TABLEBASE, params={"fen": board.fen()}, timeout=timeout)
        r.raise_for_status()
        info = r.json()
    except (requests.RequestException, ValueError):
        return None
    moves = info.get("moves") or []
    outcome = {"win": 1, "loss": -1, "draw": 0, "cursed-win": 0, "blessed-loss": 0}.get(info.get("category"))
    if not moves or outcome is None:
        return None
    best = moves[0]   # sorted best first for the side to move
    move = chess.Move.from_uci(best["uci"])
    if move not in board.legal_moves:
        return None
    sign = 1 if board.turn == chess.WHITE else -1
    distance = f", mate in {abs(info['dtm'])}" if outcome and info.get("dtm") else ""
    return move, sign * outcome * 5000.0, f"tablebase {info['category']}{distance}"


def ponder_note(last_ponder):
    """How the opponent's move compared with our pondering, for the move log (read by
    Engine/ponderstats.py): ", ponder hit #2/5 at d4" (the 2nd likeliest reply, with
    up to 5 pondered (--ponder-replies), pondered to depth 4), ", ponder miss: #7/5" (ranked 7th, not
    searched), ", ponder miss: unranked" (not among the replies considered), or ""
    when there was no pondering."""
    if last_ponder is None:
        return ""
    rank, searched, depth = last_ponder
    if rank is None:
        return ", ponder miss: unranked"
    if rank > searched or depth is None:
        return f", ponder miss: #{rank}/{searched}"
    return f", ponder hit #{rank}/{searched} at d{depth}"


def move_budget(left, increment, moves_left=MOVES_LEFT):
    """Seconds to think on this move: 1/moves_left of the clock plus the increment,
    minus network lag, never more than a third of what's left."""
    left = max(0.0, left - LAG)
    return min(left / moves_left + increment, left / 3)


class GamePlayer:
    """Plays one game on its own thread: follows the game stream, searches on our
    turn and ponders on theirs."""

    def __init__(self, li, game_id, me, args, book):
        self.li, self.game_id, self.me, self.args = li, game_id, me, args
        name, depth = args.bot
        self.bot = load_bot(name, depth=depth, device=args.device, book=book,
                            movetime=1.0)   # searches by time; each move passes its own budget
        if args.ponder_replies:
            self.bot.ponder_replies = args.ponder_replies   # default: PONDER_REPLIES in bots.py
        self.pondering = None
        self.plies = -1      # moves in the last position handled, to skip repeats (draw offers, clock updates)
        self.score = None    # White-perspective score of our last search (None: none yet, or a book move)
        self.draw_answered = None   # plies when we last answered a draw offer
        self.result = None   # (our score 1 / 0.5 / 0, opponent name, AI level or None)

    def run(self):
        """Follow the game stream until the game ends, reconnecting if it drops: the
        first event after (re)connecting is the full current state, so a move that
        never got through is searched and sent again."""
        try:
            for attempt in range(30):
                try:
                    if self.follow(first=attempt == 0):
                        return
                except requests.RequestException as e:
                    print(f"[{self.game_id}] connection lost ({e}), reconnecting")
                    self.stop_ponder()
                    self.plies = -1   # handle the current position again once back
                    time.sleep(min(1 + attempt, 10))
            print(f"[{self.game_id}] giving up after repeated connection failures")
        finally:
            self.stop_ponder()

    def follow(self, first):
        """Read the game stream; True once the game is over."""
        stream = self.li.stream(f"/api/bot/game/stream/{self.game_id}")
        full = next(e for e in stream if e)
        if first:
            self.color = chess.WHITE if full["white"].get("id") == self.me else chess.BLACK
            opponent = full["black" if self.color == chess.WHITE else "white"]
            self.opponent = opponent.get("name") or f"Stockfish level {opponent.get('aiLevel')}"
            self.ai_level = opponent.get("aiLevel")
            fen = full.get("initialFen", "startpos")
            self.start = chess.Board() if fen == "startpos" else chess.Board(fen)
            base = (full.get("clock") or {}).get("initial", 180_000) / 1000
            self.moves_left = self.args.moves_left or (MOVES_LEFT_BULLET if base < 180 else MOVES_LEFT)
            print(f"[{self.game_id}] {'White' if self.color == chess.WHITE else 'Black'} vs {self.opponent}"
                  f" ({full['speed']}, {'rated' if full['rated'] else 'casual'}): https://lichess.org/{self.game_id}")
        if not self.on_state(full["state"]):
            return True
        for event in stream:
            if event and event["type"] == "gameState" and not self.on_state(event):
                return True
        return False   # the stream ended without the game ending: reconnect

    def on_state(self, state):
        """Handle a game state; returns False once the game is over."""
        if state["status"] not in ("created", "started"):
            winner = state.get("winner")
            score = 0.5 if winner is None else 1.0 if (winner == "white") == (self.color == chess.WHITE) else 0.0
            self.result = (score, self.opponent, self.ai_level)
            print(f"[{self.game_id}] {state['status']}: {'won' if score == 1 else 'drew' if score == 0.5 else 'lost'}")
            return False
        board = self.start.copy()
        for uci in state["moves"].split():
            board.push_uci(uci)
        them = "b" if self.color == chess.WHITE else "w"
        if state.get(f"{them}draw") and self.draw_answered != len(board.move_stack):
            self.draw_answered = len(board.move_stack)
            ours = None if self.score is None else self.score * (1 if self.color == chess.WHITE else -1)
            accept = ours is not None and ours < DRAW_BELOW
            print(f"[{self.game_id}] draw offered at {'?' if ours is None else f'{ours / 100:+.2f}'} for us: "
                  f"{'accepting' if accept else 'declining'}")
            self.try_post(f"/api/bot/game/{self.game_id}/draw/{'yes' if accept else 'no'}")
        if state.get(f"{them}takeback"):
            self.try_post(f"/api/bot/game/{self.game_id}/takeback/no")
        if board.is_game_over() or len(board.move_stack) == self.plies:
            return True
        self.plies = len(board.move_stack)
        if board.turn == self.color:
            self.play(board, state)
        elif self.pondering is None and not self.args.no_ponder:
            self.pondering = self.bot.ponder(board)
        return True

    def play(self, board, state):
        if self.pondering is not None:
            self.pondering.finish(board)
            self.pondering = None
        mine = "w" if self.color == chess.WHITE else "b"
        budget = move_budget(state[f"{mine}time"] / 1000, state[f"{mine}inc"] / 1000, self.moves_left)
        start = time.monotonic()
        found = None if self.args.no_tablebase else tablebase_move(board, timeout=min(1.5, budget / 2))
        if found:
            move, score, how = found
        else:
            try:
                move, score = self.bot.search(board, movetime=budget)
            except SearchAborted:
                return
            except Exception as e:   # never stop playing over a search bug: retry once without pondered results
                print(f"[{self.game_id}] search failed ({e!r}), searching again without pondered results")
                self.bot.pondered.clear()
                move, score = self.bot.search(board, movetime=max(0.2, budget - (time.monotonic() - start)))
            how = "book" if self.bot.last_book else f"depth {self.bot.last_depth}{ponder_note(self.bot.last_ponder)}"
        if score is not None:
            self.score = score
        print(f"[{self.game_id}] {board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
              f"{board.san(move)}  ({how}, {time.monotonic() - start:.2f}s of {budget:.2f}s)")
        self.try_post(f"/api/bot/game/{self.game_id}/move/{move.uci()}")

    def try_post(self, path):
        try:
            self.li.post(path)
        except requests.HTTPError as e:   # e.g. the game ended while we were thinking
            print(f"[{self.game_id}] {e}")

    def stop_ponder(self):
        if self.pondering is not None:
            self.pondering.finish()
            self.pondering = None


class BotRunner:
    def __init__(self, li, args):
        self.li, self.args = li, args
        account = li.get("/api/account")
        self.me = account["id"]
        if account.get("title") != "BOT":
            raise SystemExit(f"{account['username']} isn't a BOT account yet: run `lichess.py upgrade` first")
        self.book = (OpeningBook(os.path.join(ROOT, "Models", "Books", args.book), max_plies=args.book_plies)
                     if args.book else None)
        self.lock = threading.Lock()
        self.games = {}   # game id -> GamePlayer
        if args.challenge == ["auto"]:   # online bots rated --opponent-rating at this clock's speed
            speed, found = find_opponents(li, args.clock, args.opponent_rating, args.opponent_count, {self.me})
            args.challenge = [name for _, name in found]
            print(f"challenging {len(found)} online bots rated {args.opponent_rating[0]:g}-"
                  f"{args.opponent_rating[1]:g} in {speed}: " + ", ".join(f"{n} ({r})" for r, n in found))
        # Opponents to challenge when idle, in turn: ("ai", level) or ("user", name)
        self.opponents = ([("ai", level) for level in args.stockfish or []]
                          + [("user", name) for name in args.challenge or []])
        self.turn = 0          # index of the next opponent
        self.pending = None    # (challenge or game id, opponent, sent at) while waiting for an answer
        self.accepted = {}     # incoming challenge id -> accepted at, until its game starts (holds a slot)
        self.refusals = defaultdict(int)   # opponent -> declines / no answers in a row
        self.challenged = set()            # game ids from our own challenges
        self.started = 0                   # our challenges that became games
        self.results = defaultdict(list)   # opponent name -> our scores, from our own challenges
        print(f"Logged in as {account['username']}: https://lichess.org/@/{account['username']}")

    def run(self):
        # Resume games already in progress (e.g. after a restart) before challenging
        # anyone, so a new challenge can't start a second game alongside them
        for game in self.li.get("/api/account/playing").get("nowPlaying", []):
            print(f"resuming game {game['gameId']} in progress")
            self.start_game(game["gameId"])
        for path in (STOP_FILE, PAUSE_FILE):   # leftovers from an earlier run
            if os.path.exists(path):
                os.remove(path)
        threading.Thread(target=self.challenge_when_idle, daemon=True).start()
        threading.Thread(target=self.watch_controls, daemon=True).start()
        while True:
            try:
                for event in self.li.stream("/api/stream/event"):
                    if event:
                        self.on_event(event)
            except requests.RequestException as e:
                print(f"event stream dropped ({e}), reconnecting")
                time.sleep(5)

    def on_event(self, event):
        kind = event["type"]
        if kind in ("challengeDeclined", "challengeCanceled"):
            c = event["challenge"]
            with self.lock:
                if self.pending and self.pending[0] == c["id"]:
                    opponent = self.pending[1]
                    self.refusals[opponent] += 1
                    self.pending = None
                    print(f"{label(opponent)} declined: {c.get('declineReason') or 'canceled'}")
        elif kind == "challenge":
            c = event["challenge"]
            if c.get("direction") == "out" or c["challenger"]["id"] == self.me:
                return
            with self.lock:   # decide and claim the slot together, so the challenge loop can't take it
                reason = self.decline_reason(c)
                if not reason:
                    self.accepted[c["id"]] = time.monotonic()
            who = f"{c['challenger']['name']} ({c['speed']}, {'rated' if c['rated'] else 'casual'})"
            try:
                if reason:
                    self.li.post(f"/api/challenge/{c['id']}/decline", {"reason": reason})
                    print(f"declined {who}: {reason}")
                else:
                    try:
                        self.li.post(f"/api/challenge/{c['id']}/accept")
                    except requests.RequestException:
                        with self.lock:
                            self.accepted.pop(c["id"], None)
                        raise
                    print(f"accepted {who}")
            except requests.RequestException as e:
                print(f"challenge {c['id']}: {e}")
        elif kind == "gameStart":
            self.start_game(event["game"]["id"])

    def start_game(self, game_id):
        """Play game_id on its own thread, unless it's already being played."""
        with self.lock:
            if game_id in self.games:
                return
            self.accepted.pop(game_id, None)   # a challenge's game has its id
            player = GamePlayer(self.li, game_id, self.me, self.args, self.book)
            self.games[game_id] = player
            if self.pending and self.pending[0] == game_id:
                self.refusals[self.pending[1]] = 0
                self.pending = None
                self.challenged.add(game_id)
                self.started += 1
        threading.Thread(target=self.play, args=(player,), daemon=True).start()

    def decline_reason(self, c):
        """A Lichess decline reason key, or None to accept. Called with self.lock held."""
        a, tc = self.args, c["timeControl"]
        if c["variant"]["key"] != "standard":
            return "standard"
        if tc.get("type") != "clock":
            return "timecontrol"
        if tc["limit"] + 40 * tc["increment"] < a.min_base:   # estimated game length, as Lichess does
            return "toofast"
        if tc["limit"] > a.max_base:
            return "tooslow"
        if tc["increment"] == 0 and not a.allow_no_increment:
            return "timecontrol"   # the budget assumes an increment; with ~0.2 s/move of lag it flags
        if c["rated"] and a.casual_only:
            return "casual"
        if a.no_bots and c["challenger"].get("title") == "BOT":
            return "nobot"
        if len(self.games) + len(self.accepted) + (self.pending is not None) >= a.max_games:
            return "later"
        if os.path.exists(PAUSE_FILE) or os.path.exists(STOP_FILE):
            return "later"
        return None

    def watch_controls(self):
        """Pause / stop between games (scripts/lichess-bot.sh): with STOP_FILE, exit as soon
        as no game is in progress or about to start, canceling a challenge still unanswered."""
        paused = False
        while True:
            now_paused = os.path.exists(PAUSE_FILE)
            if now_paused != paused:
                print("paused: no new games until resumed" if now_paused else "resumed")
                paused = now_paused
            if os.path.exists(STOP_FILE):
                with self.lock:
                    pending = self.pending
                    idle = not self.games and not self.accepted and (pending is None or pending[0] is not None)
                if idle:
                    if pending is not None:
                        self.cancel(pending[0])
                    print("stopped between games")
                    self.report()
                    os.remove(STOP_FILE)
                    os._exit(0)
            time.sleep(0.5)

    def play(self, player):
        try:
            player.run()
        except Exception as e:
            print(f"[{player.game_id}] failed: {e!r}")
        with self.lock:
            del self.games[player.game_id]
            if player.result and player.game_id in self.challenged:
                self.results[player.result[1]].append(player.result[0])

    def challenge_when_idle(self):
        a = self.args
        if not self.opponents:
            return
        base, inc = a.clock
        while True:
            opponent = None
            with self.lock:
                if self.pending and time.monotonic() - self.pending[2] > 60:   # no answer: offline or ignoring us
                    stale, self.pending = self.pending, None
                    self.refusals[stale[1]] += 1
                    print(f"{label(stale[1])} didn't answer; canceling")
                    threading.Thread(target=self.cancel, args=(stale[0],), daemon=True).start()
                willing = [o for o in self.opponents if self.refusals[o] < 2]
                if sum(len(v) for v in self.results.values()) >= a.games or not willing:
                    print("challenge list done" if willing else "every opponent declined",
                          "- still online for challenges")
                    self.report()
                    return
                for cid, at in list(self.accepted.items()):   # accepted but never started (e.g. aborted)
                    if time.monotonic() - at > 60:
                        del self.accepted[cid]
                held = os.path.exists(PAUSE_FILE) or os.path.exists(STOP_FILE)
                if not held and not self.games and not self.accepted and not self.pending and self.started < a.games:
                    while self.refusals[self.opponents[self.turn % len(self.opponents)]] >= 2:
                        self.turn += 1
                    opponent = self.opponents[self.turn % len(self.opponents)]
                    self.turn += 1
                    self.pending = (None, opponent, time.monotonic())   # holds the slot while we ask
            if opponent:
                kind, who = opponent
                clock = {"clock.limit": int(base), "clock.increment": int(inc), "color": "random"}
                try:
                    if kind == "ai":
                        reply = self.li.post("/api/challenge/ai", {"level": who, **clock})
                    else:
                        reply = self.li.post(f"/api/challenge/{who}", {"rated": str(a.rated).lower(), **clock})
                    with self.lock:
                        if self.pending and self.pending[1] == opponent and self.pending[0] is None:
                            self.pending = (reply["id"], opponent, time.monotonic())
                        if reply["id"] in self.games:   # the game started before we got here
                            self.pending = None
                            self.challenged.add(reply["id"])
                            self.started += 1
                    print(f"challenged {label(opponent)} ({base / 60:g}+{inc:g}"
                          f"{', rated' if a.rated and kind == 'user' else ''}), game {self.started + 1}/{a.games}")
                except requests.RequestException as e:
                    print(f"couldn't challenge {label(opponent)}: {e}")
                    with self.lock:
                        self.pending = None
                        self.refusals[opponent] += 1
                    time.sleep(5)
            time.sleep(2)

    def cancel(self, challenge_id):
        try:
            self.li.post(f"/api/challenge/{challenge_id}/cancel")
        except requests.RequestException:
            pass

    def report(self):
        if not self.results:
            return
        print("\nSlackFish's score against the opponents it challenged:")
        for name in sorted(self.results):
            s = self.results[name]
            print(f"  {name}: {sum(s):g}/{len(s)}  (+{s.count(1.0)} ={s.count(0.5)} -{s.count(0.0)})")


def speed_of(clock):
    """Lichess's speed for a clock: estimated seconds a side, base + 40 x increment."""
    base, inc = clock
    est = base + 40 * inc
    return "ultraBullet" if est < 30 else "bullet" if est < 180 else "blitz" if est < 480 else "rapid" if est < 1500 else "classical"


def find_opponents(li, clock, rating, count=20, exclude=()):
    """Online bots rated within rating (low, high) at clock's speed, with 50+ games and an
    established rating, strongest first: names for --challenge."""
    speed = speed_of(clock)
    low, high = rating
    rows = []
    for u in li.get_ndjson("/api/bot/online", {"nb": 300}):
        perf = u.get("perfs", {}).get(speed, {})
        if (u["id"] in exclude or u.get("tosViolation") or u.get("disabled") or perf.get("prov")
                or perf.get("games", 0) < 50 or not low <= perf.get("rating", 0) < high):
            continue
        rows.append((perf["rating"], u["username"]))
    rows.sort(reverse=True)
    return speed, rows[:count]


def parse_range(spec):
    low, _, high = spec.partition("-")
    return float(low), float(high)


def label(opponent):
    kind, who = opponent
    return f"Stockfish level {who}" if kind == "ai" else who


def parse_bot(spec):
    name, _, depth = spec.partition(":")
    if name not in BOTS:
        raise argparse.ArgumentTypeError(f"unknown bot {name!r}; available: {', '.join(BOTS)}")
    return name, int(depth or 3)


def parse_clock(spec):
    minutes, _, inc = spec.partition("+")
    return float(minutes) * 60, float(inc or 0)


def main():
    parser = argparse.ArgumentParser(description="Run SlackFish as a Lichess bot.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upgrade", help="turn the token's account into a BOT account (permanent; no games played yet)")
    p = sub.add_parser("play", help="go online: accept challenges, optionally play the Lichess AI")
    p.add_argument("--bot", type=parse_bot, default=("t2-sg-wsd-lichessgames-8.87m", 3),
                   help="NAME[:DEPTH]; DEPTH is when a pondered reply is played at once (default t2-sg-wsd-lichessgames-8.87m:3)")
    p.add_argument("--book", nargs="?", const="gm2001.bin", help="opening book in Models/Books (default with no name: gm2001.bin)")
    p.add_argument("--book-plies", type=int, default=16,
                   help="leave the book after this many plies at most (default 16; deep engine books like Cerebellum go 30+)")
    p.add_argument("--max-games", type=int, default=1, help="games at once (default 1)")
    p.add_argument("--min-base", type=float, default=30,
                   help="decline games estimated under this many seconds a side, base + 40 x increment (default 30)")
    p.add_argument("--max-base", type=float, default=1800, help="decline clocks over this many seconds (default 1800)")
    p.add_argument("--allow-no-increment", action="store_true",
                   help="accept clocks without increment (e.g. 1+0, 5+0), declined by default since the bot can flag")
    p.add_argument("--casual-only", action="store_true", help="decline rated challenges")
    p.add_argument("--no-bots", action="store_true", help="decline challenges from other bots")
    p.add_argument("--no-ponder", action="store_true", help="don't think on the opponent's time")
    p.add_argument("--ponder-replies", type=int,
                   help=f"opponent replies to ponder together; fewer get deeper (default {PONDER_REPLIES}). "
                        "Engine/ponderstats.py compares settings")
    p.add_argument("--no-tablebase", action="store_true",
                   help="search endgames too, instead of asking Lichess's tablebase with 7 pieces or fewer")
    p.add_argument("--moves-left", type=float,
                   help=f"think 1/N of the remaining clock (plus the increment) per move; higher = faster "
                        f"(default: {MOVES_LEFT_BULLET} for clocks under 3 minutes, else {MOVES_LEFT})")
    p.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    p.add_argument("--stockfish", type=lambda s: [int(x) for x in s.split(",")],
                   help="challenge the Lichess AI at these levels (1-8, e.g. 2,3,4) when idle")
    p.add_argument("--challenge", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="challenge these Lichess users (usually bots) when idle, e.g. maia1,GarboBot; "
                        "'auto': online bots rated --opponent-rating at --clock's speed")
    p.add_argument("--opponent-rating", type=parse_range, default=(1700.0, 2100.0), metavar="MIN-MAX",
                   help="with --challenge auto: their rating at this speed (default 1700-2100)")
    p.add_argument("--opponent-count", type=int, default=20,
                   help="with --challenge auto: bots to take turns with, strongest first (default 20)")
    p.add_argument("--games", type=int, default=2,
                   help="with --stockfish / --challenge: games to play in total (default 2)")
    p.add_argument("--clock", type=parse_clock, default=(180.0, 2.0),
                   help="with --stockfish / --challenge: MINUTES+INCREMENT (default 3+2)")
    p.add_argument("--rated", action="store_true", help="with --challenge: rated games (default casual)")
    p.add_argument("--casual", action="store_true", help="with --challenge: casual games even if --rated is given "
                                                         "(e.g. after a preset that adds --rated)")
    o = sub.add_parser("opponents", help="list online bots to challenge (what --challenge auto picks)")
    o.add_argument("--clock", type=parse_clock, default=(180.0, 2.0), help="MINUTES+INCREMENT (default 3+2)")
    o.add_argument("--rating", type=parse_range, default=(1700.0, 2100.0), metavar="MIN-MAX",
                   help="their rating at that speed (default 1700-2100)")
    o.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if getattr(args, "casual", False):
        args.rated = False

    li = Lichess(read_token())
    if args.command == "opponents":
        speed, found = find_opponents(li, args.clock, args.rating, args.count)
        print(f"{len(found)} online bots rated {args.rating[0]:g}-{args.rating[1]:g} in {speed}:")
        for rating, name in found:
            print(f"  {rating}  {name}")
        print(",".join(name for _, name in found))
        return
    if args.command == "upgrade":
        account = li.get("/api/account")
        if account.get("title") == "BOT":
            print(f"{account['username']} is already a BOT account")
            return
        li.post("/api/bot/account/upgrade")
        print(f"{account['username']} is now a BOT account")
        return
    if args.stockfish and not all(1 <= level <= 8 for level in args.stockfish):
        raise SystemExit("--stockfish levels are 1-8")
    runner = BotRunner(li, args)
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        runner.report()


if __name__ == "__main__":
    main()
