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
increment per move (--moves-left; the budget is in Engine/clock.py), searching by time
with no depth limit. On the opponent's
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

Network trouble: a game whose connection drops keeps reconnecting for as long as it
could still be going (our clock, plus GIVE_UP_MARGIN). After any connection error it
starts no game for NETWORK_QUIET seconds (5 min): no challenges, and incoming ones
declined "later", so a flaky connection can't start games it then loses on time (on
2026-11-07 DNS failures did that: games started in brief working windows, played
their instant book moves, and flagged). A game abandoned for the connection ends the
challenge series for the run. A rejected token (401: revoked) exits at once. Log
lines start with the date and time. Anything else that goes wrong in a game (a search
error, an unexpected game state) never leaves it unplayed: a search that fails twice
plays a legal move anyway, any other error re-reads the game and carries on, and a game
where the search failed ends the challenge series.

Every game is saved to logs/games/GAMEID.pgn when it ends (or is abandoned), with
what the bot knew at each of its moves as a comment: its score (centipawns, White's
point of view, or book / tablebase), depth, ponder result, seconds used and the clock.
Engine/mine_hard.py runs Stockfish over them to find the moves it got wrong.
"""
import argparse
import faulthandler
import signal
import json
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone

import chess
import chess.pgn
import requests

from bots import BOTS, PONDER_REPLIES, ROOT, load_bot, open_books
from clock import MOVES_LEFT, MOVES_LEFT_BULLET, move_budget, moves_left_for, parse_clock, speed_of

API = "https://lichess.org"
TOKEN_FILE = os.path.join(ROOT, ".lichess_token")
DRAW_OFFER_EVERY = 10   # plies between our draw offers, which we make when the tablebase says it's a draw
DRAW_BELOW = 10   # accept a draw offer when our last eval is under this many centipawns (+0.10) for us
TABLEBASE = "https://tablebase.lichess.ovh/standard"   # Lichess's free endgame tablebase (up to 7 pieces)
TABLEBASE_PIECES = 7
# Control files (scripts/lichess-bot.sh pause / resume / stop): checked twice a second
PAUSE_FILE = os.path.join(ROOT, "logs", "lichess.pause")   # finish games in progress, then start none
STOP_FILE = os.path.join(ROOT, "logs", "lichess.stop")     # ... and exit once none is in progress
ANSWER_WAIT = 30   # seconds an idle, online opponent gets to accept a challenge before we cancel it
NETWORK_QUIET = 300   # seconds without a connection error before challenging or accepting a game
GIVE_UP_MARGIN = 120   # keep reconnecting to a game this long past our clock running out
GAMES_DIR = os.path.join(ROOT, "logs", "games")   # one PGN per game, for Engine/mine_hard.py


def read_token():
    token = os.environ.get("LICHESS_TOKEN")
    if not token and os.path.exists(TOKEN_FILE):
        token = open(TOKEN_FILE).read().strip()
    if not token:
        raise SystemExit(f"No Lichess token: set LICHESS_TOKEN or write it to {TOKEN_FILE} (see --help)")
    return token


GAME_STREAM_TIMEOUT = 20   # seconds of silence (Lichess sends a keep-alive line every few seconds) before a game stream is dropped and reconnected


class Lichess:
    """The few Lichess API calls a bot needs."""

    def __init__(self, token):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.token = token
        self.fast_session = self.new_session()   # moves only: replaced after a stall, so the streams' connections are left alone
        self.last_error = None   # time.monotonic() of the last failed connection

    def new_session(self):
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {self.token}"
        return s

    def quiet_for(self):
        """Seconds since the last connection error (inf if none yet)."""
        return float("inf") if self.last_error is None else time.monotonic() - self.last_error

    def _request(self, method, path, session=None, **kwargs):
        """One request: records connection errors (see quiet_for) and exits on 401."""
        try:
            r = (session or self.session).request(method, API + path, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            self.last_error = time.monotonic()
            raise
        if r.status_code == 401:
            # The token was revoked or is wrong: nothing can work, so stop rather than retry forever
            print(f"Lichess rejected the token (401 on {path}): revoked or wrong. Exiting.", flush=True)
            os._exit(1)
        return r

    def get(self, path):
        r = self._request("GET", path, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_ndjson(self, path, params=None):
        r = self._request("GET", path, params=params, headers={"Accept": "application/x-ndjson"}, timeout=30)
        r.raise_for_status()
        return [json.loads(line) for line in r.text.splitlines() if line.strip()]

    def post(self, path, data=None, fast=False):
        """fast (moves): a move normally takes ~0.1 s, so wait only 3 s for the reply and retry at once on a new
        connection; a stalled one cost 45 s and the game on time in bullet. Otherwise: POST, retrying dropped connections and server errors (Lichess closes idle
        keep-alive connections, so the first request after a pause can fail; an
        unsent move would otherwise lose on time) and waiting out rate limits."""
        tries = 12 if fast else 6
        for attempt in range(tries):
            try:
                if fast:
                    r = self._request("POST", path, session=self.fast_session, data=data, timeout=(2, 3))
                else:
                    r = self._request("POST", path, data=data, timeout=15)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == tries - 1:
                    raise
                print(f"POST {path} failed ({e.__class__.__name__}), retrying")
                if fast:
                    self.fast_session = self.new_session()
                else:
                    time.sleep(0.25 * 2 ** attempt)
                continue
            if r.status_code == 429:
                print("rate limited by Lichess, waiting a minute")
                time.sleep(60)
            elif r.status_code >= 500 and attempt < tries - 1:
                time.sleep(0.25 * 2 ** attempt)
            else:
                break
        if not r.ok:
            raise requests.HTTPError(f"{r.status_code} {path}: {r.text.strip()}", response=r)
        return r.json() if r.content else None

    def stream(self, path, read_timeout=90):
        """Yield each event of an ndjson stream, or None for Lichess's blank
        keep-alive lines (every few seconds), so a reader can check for stopping.
        read_timeout: seconds of silence before the stream counts as dead (a game stream
        uses 20: a stalled connection once went unnoticed for 90 s, longer than a bullet game)."""
        with self._request("GET", path, stream=True, timeout=(30, read_timeout)) as r:
            r.raise_for_status()
            try:
                for line in r.iter_lines():
                    yield json.loads(line) if line else None
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                self.last_error = time.monotonic()   # dropped mid-stream
                raise


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
    Engine/ponderstats.py): ", ponder hit #2/3 at d4" (the 2nd likeliest reply, with
    3 pondered (--ponder-replies), pondered to depth 4), ", ponder miss: #7/3" (ranked 7th, not
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


def fallback_move(board):
    """A legal move without the neural network, for when the search fails: a mate in one if
    there is one, else the most valuable capture, else the first legal move."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    best, best_key = None, None
    for move in board.legal_moves:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return move
        captured = board.piece_type_at(move.to_square) or (chess.PAWN if board.is_en_passant(move) else None)
        key = values[captured] if captured else -1
        if best_key is None or key > best_key:
            best, best_key = move, key
    return best


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
        self.draw_offered = -DRAW_OFFER_EVERY   # plies when we last offered one
        self.result = None   # (our score 1 / 0.5 / 0, opponent name, AI level or None)
        # For the game's PGN (save_pgn): the game stream's first event, the moves so far,
        # (wtime, btime) in ms by number of moves played, and our notes by ply: (uci, comment)
        self.full, self.moves, self.clocks, self.notes = None, "", {}, {}
        self.saved = False   # whether save_pgn ran
        self.abandoned = False   # gave up on the connection: the game was most likely lost on time
        self.search_failed = False   # a search failed twice and a fallback move was played

    def run(self):
        """Follow the game stream until the game ends, reconnecting if it drops: the
        first event after (re)connecting is the full current state, so a move that
        never got through is searched and sent again. Keeps trying for as long as the
        game could still be going (our clock, plus 2 minutes), then gives up."""
        self.outage = None   # time.monotonic() when the current run of failures began
        attempt = 0
        try:
            while True:
                try:
                    if self.follow(first=self.full is None):
                        return
                except requests.RequestException as e:
                    now = time.monotonic()
                    self.outage = self.outage or now
                    if now - self.outage > self.time_left() + GIVE_UP_MARGIN:
                        print(f"[{self.game_id}] giving up: no connection for {now - self.outage:.0f} s")
                        self.abandoned = True
                        return
                    print(f"[{self.game_id}] connection lost ({e}), reconnecting")
                    self.stop_ponder()
                    self.plies = -1   # handle the current position again once back
                    attempt += 1
                    time.sleep(min(attempt, 10))
                except Exception as e:
                    # Not the network: a bug or a GPU error. Never leave the game unplayed: read it
                    # again from the stream and handle the current position afresh. Once the game is
                    # over, the stream's first event says so and follow() returns True.
                    print(f"[{self.game_id}] error ({e!r}), reading the game again")
                    traceback.print_exc(file=sys.stdout)
                    self.stop_ponder()
                    self.plies = -1
                    attempt += 1
                    time.sleep(min(attempt, 10))
        finally:
            self.stop_ponder()
            if not self.saved:   # ended abnormally: save what we have, result "*"
                self.save_pgn(None)

    def time_left(self):
        """Seconds on our clock at the last update (10 minutes if none yet)."""
        if not self.clocks or self.full is None:
            return 600
        wtime, btime = self.clocks[max(self.clocks)]
        return (wtime if self.color == chess.WHITE else btime) / 1000

    def follow(self, first):
        """Read the game stream; True once the game is over."""
        stream = self.li.stream(f"/api/bot/game/stream/{self.game_id}", read_timeout=GAME_STREAM_TIMEOUT)
        full = next(e for e in stream if e)
        self.outage = None   # connected again
        if first:
            self.full = full
            self.color = chess.WHITE if full["white"].get("id") == self.me else chess.BLACK
            opponent = full["black" if self.color == chess.WHITE else "white"]
            self.opponent = opponent.get("name") or f"Stockfish level {opponent.get('aiLevel')}"
            self.ai_level = opponent.get("aiLevel")
            fen = full.get("initialFen", "startpos")
            self.start = chess.Board() if fen == "startpos" else chess.Board(fen)
            clock = full.get("clock") or {}
            self.moves_left = self.args.moves_left or moves_left_for(
                (clock.get("initial", 180_000) / 1000, clock.get("increment", 2_000) / 1000))
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
        self.moves = state.get("moves", self.moves)
        self.clocks[self.moves.count(" ") + 1 if self.moves else 0] = (state.get("wtime"), state.get("btime"))
        if state["status"] not in ("created", "started"):
            winner = state.get("winner")
            score = 0.5 if winner is None else 1.0 if (winner == "white") == (self.color == chess.WHITE) else 0.0
            self.result = (score, self.opponent, self.ai_level)
            print(f"[{self.game_id}] {state['status']}: {'won' if score == 1 else 'drew' if score == 0.5 else 'lost'}")
            self.save_pgn(state)
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
        entered = time.monotonic()
        if self.pondering is not None:
            self.pondering.finish(board)
            self.pondering = None
        if time.monotonic() - entered > 1:
            print(f"[{self.game_id}] slow: stopping the ponder took {time.monotonic() - entered:.1f}s")
        mine = "w" if self.color == chess.WHITE else "b"
        budget = move_budget(state[f"{mine}time"] / 1000, state[f"{mine}inc"] / 1000, self.moves_left)
        start = time.monotonic()
        found = None if self.args.no_tablebase else tablebase_move(board, timeout=min(1.5, budget / 2))
        if found:
            move, score, how = found
        else:
            move, score, how = self.search(board, budget, start)
        if score is not None:
            self.score = score
        took = time.monotonic() - start
        print(f"[{self.game_id}] {board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
              f"{board.san(move)}  ({how}, {took:.2f}s of {budget:.2f}s)")
        posted = time.monotonic()
        self.try_post(f"/api/bot/game/{self.game_id}/move/{move.uci()}")
        plies = len(board.move_stack) + 1   # with our move played
        if how in ("tablebase draw", "tablebase blessed-loss") and plies - self.draw_offered >= DRAW_OFFER_EVERY:
            self.draw_offered = plies
            print(f"[{self.game_id}] tablebase draw: offering a draw")
            self.try_post(f"/api/bot/game/{self.game_id}/draw/yes")
        if time.monotonic() - posted > 1:
            print(f"[{self.game_id}] slow: sending the move took {time.monotonic() - posted:.1f}s")
        # This move's PGN comment, the log line with the score added (read by Engine/mine_hard.py),
        # e.g. "score +35, depth 6, ponder hit #2/3 at d4, 3.10s of 12.50s" or "score book, ..."
        try:
            searched = score is not None and not found
            self.notes[plies - 1] = (move.uci(), f"score {f'{score:+.0f}, ' if searched else ''}"
                                                 f"{how}, {took:.2f}s of {budget:.2f}s")
        except Exception as e:   # a log must never stop play
            print(f"[{self.game_id}] couldn't note the move for the PGN ({e!r})")

    def search(self, board, budget, start):
        """(move, score, log note) from the bot's search. Never fails to give a move: a
        failed search is tried once more without pondered results, and if that fails too
        (e.g. a GPU error) a legal move is played anyway (score None) rather than none."""
        for attempt in range(2):
            try:
                if attempt:
                    self.bot.pondered.clear()
                left = budget - (time.monotonic() - start)
                move, score = self.bot.search(board, movetime=budget if attempt == 0 else max(0.2, left))
                if move not in board.legal_moves:
                    raise ValueError(f"search returned {move}, not a legal move")
                how = ("book" if self.bot.last_book
                       else f"depth {self.bot.last_depth}{ponder_note(self.bot.last_ponder)}")
                return move, score, how
            except Exception as e:
                print(f"[{self.game_id}] search failed ({e!r})"
                      + (", searching again without pondered results" if attempt == 0 else ""))
                traceback.print_exc(file=sys.stdout)
        self.search_failed = True
        move = fallback_move(board)
        print(f"[{self.game_id}] playing {board.san(move)} without a search")
        return move, None, "fallback: search failed"

    def save_pgn(self, end):
        """Write the game to GAMES_DIR/GAMEID.pgn with our notes (see play) and the clocks
        as comments. end: the final game state, or None if the game ended abnormally
        (result "*"). Rewrites the file whole (via a temporary file), so a game resumed
        after a restart replaces it, with notes only for the moves made since. Never
        raises: a log must never stop play."""
        self.saved = True   # tried: a failure isn't retried, so it's reported once
        try:
            if self.full is None or not self.moves:   # never joined, or no move played (aborted)
                return
            full, game = self.full, chess.pgn.Game()
            h = game.headers
            h["Event"] = f"{'Rated' if full.get('rated') else 'Casual'} {full.get('speed', '')} game"
            h["Site"] = f"https://lichess.org/{self.game_id}"
            created = full.get("createdAt")
            h["Date"] = (datetime.fromtimestamp(created / 1000, timezone.utc) if created
                         else datetime.now(timezone.utc)).strftime("%Y.%m.%d")
            for side in ("white", "black"):
                player = full.get(side) or {}
                h[side.title()] = player.get("name") or f"Stockfish level {player.get('aiLevel')}"
                if player.get("rating"):
                    h[f"{side.title()}Elo"] = str(player["rating"])
            winner = end and end.get("winner")
            unfinished = end is None or end["status"] in ("aborted", "noStart")
            h["Result"] = ("*" if unfinished and not winner else "1-0" if winner == "white"
                           else "0-1" if winner == "black" else "1/2-1/2")
            clock = full.get("clock") or {}
            h["TimeControl"] = f"{clock['initial'] // 1000}+{clock['increment'] // 1000}" if clock else "-"
            h["Rated"] = "Yes" if full.get("rated") else "No"
            h["Termination"] = end["status"] if end else "unfinished"
            h["Bot"] = "{}:{}".format(*self.args.bot)
            if self.start != chess.Board():
                game.setup(self.start)
            node = game
            for i, uci in enumerate(self.moves.split()):
                node = node.add_variation(chess.Move.from_uci(uci))
                note = self.notes.get(i)
                if note and note[0] == uci:
                    node.comment = note[1]
                # The mover's clock once it moved; it holds through the reply, if we missed that state
                white = (i % 2 == 0) == (self.start.turn == chess.WHITE)
                left = (self.clocks.get(i + 1) or self.clocks.get(i + 2) or (None, None))[0 if white else 1]
                if left is not None:
                    node.set_clock(left / 1000)
            os.makedirs(GAMES_DIR, exist_ok=True)
            path = os.path.join(GAMES_DIR, f"{self.game_id}.pgn")
            with open(path + ".tmp", "w") as f:
                f.write(str(game) + "\n")
            os.replace(path + ".tmp", path)
        except Exception as e:
            print(f"[{self.game_id}] couldn't save the PGN ({e!r})")

    def try_post(self, path):
        try:
            self.li.post(path, fast=True)
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
        self.book = self.open_book()
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
        self.free, self.free_at, self.waiting_note = None, -1e9, 0.0
        self.refusals = defaultdict(int)   # opponent -> declines / no answers in a row
        self.challenged = set()            # game ids from our own challenges
        self.started = 0                   # our challenges that became games
        self.results = defaultdict(list)   # opponent name -> our scores, from our own challenges
        self.stream_up = False             # the event stream is connected
        self.halted = None                 # why the challenge series stopped early, if it did
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
                    self.stream_up = True
                    if event:
                        self.on_event(event)
            except requests.RequestException as e:
                self.stream_up = False
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
            player = GamePlayer(self.li, game_id, self.me, self.args, self.open_book())
            self.games[game_id] = player
            if self.pending and self.pending[0] == game_id:
                self.refusals[self.pending[1]] = 0
                self.pending = None
                self.challenged.add(game_id)
                self.started += 1
        threading.Thread(target=self.play, args=(player,), daemon=True).start()

    def open_book(self):
        """The opening book(s), opened afresh for each game so a book file rewritten while the bot runs
        (see Engine/make_book.py; write to a temp file and rename) is used from the next game on."""
        if not self.args.book:
            return None
        return open_books([os.path.join(ROOT, "Models", "Books", name) for name in self.args.book.split(",")],
                          max_plies=self.args.book_plies)

    def decline_reason(self, c):
        """A Lichess decline reason key, or None to accept. Called with self.lock held."""
        a, tc = self.args, c["timeControl"]
        if c["variant"]["key"] not in ("standard", "fromPosition"):   # fromPosition: odds games from a FEN
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
        if self.li.quiet_for() < NETWORK_QUIET:   # a game started now could be lost on time
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
            if (player.abandoned or player.search_failed) and self.opponents and not self.halted:
                # Don't start more games on a connection, or a GPU, that just cost one
                self.halted = (f"lost the connection during game {player.game_id}" if player.abandoned
                               else f"the search failed during game {player.game_id}")
                print(f"stopping the challenge series: {self.halted}. Restart it once the network is stable.")

    def free_users(self):
        """Names of the listed opponents that are online and not in a game (None if Lichess can't say).
        Busy or offline bots can't accept, and waiting out their silence cost a minute and a strike each."""
        if time.monotonic() - self.free_at < 15:
            return self.free
        names = [who for kind, who in self.opponents if kind == "user"]
        free = None
        if names:
            try:
                r = self.li._request("GET", "/api/users/status", params={"ids": ",".join(names)}, timeout=15)
                r.raise_for_status()
                up = {u["name"].lower() for u in r.json() if u.get("online") and not u.get("playing")}
                free = {n for n in names if n.lower() in up}
            except (requests.RequestException, ValueError, KeyError):
                free = None
        self.free, self.free_at = free, time.monotonic()
        return free

    def challenge_when_idle(self):
        a = self.args
        if not self.opponents:
            return
        base, inc = a.clock
        holding = False   # printed that we're waiting for a stable connection
        while True:
            opponent = None
            if self.halted:
                self.report()
                return
            unstable = self.li.quiet_for() < NETWORK_QUIET or not self.stream_up
            if unstable != holding:
                holding = unstable
                print(f"connection unstable: no new challenges until {NETWORK_QUIET // 60} minutes without errors"
                      if unstable else "connection stable again: resuming challenges")
            free = self.free_users() if not self.pending and not self.games and not unstable else None
            with self.lock:
                if self.pending and time.monotonic() - self.pending[2] > ANSWER_WAIT:   # no answer: offline or ignoring us
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
                if (not held and not unstable and not self.games and not self.accepted and not self.pending
                        and self.started < a.games):
                    for _ in range(len(self.opponents)):   # next willing opponent that is online and not in a game
                        cand = self.opponents[self.turn % len(self.opponents)]
                        self.turn += 1
                        if self.refusals[cand] < 2 and (cand[0] == "ai" or free is None or cand[1] in free):
                            opponent = cand
                            break
                    if opponent:
                        self.pending = (None, opponent, time.monotonic())   # holds the slot while we ask
                    elif time.monotonic() - self.waiting_note > 600:
                        self.waiting_note = time.monotonic()
                        print("no listed opponent is online and free; waiting")
            if opponent:
                kind, who = opponent
                clock = {"clock.limit": int(base), "clock.increment": int(inc), "color": a.color}
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
                    # rate limits (429) and network errors say nothing about the opponent: wait and try again
                    transient = not isinstance(e, requests.HTTPError) or e.response is None or e.response.status_code == 429
                    with self.lock:
                        self.pending = None
                        if not transient:
                            self.refusals[opponent] += 1
                    time.sleep(120 if transient else 5)
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


class Timestamped:
    """stdout with the time at the start of each line (the log had none)."""

    def __init__(self, out):
        self.out, self.line_start, self.lock = out, True, threading.Lock()

    def write(self, text):
        with self.lock:
            for part in text.splitlines(keepends=True):
                if self.line_start:
                    self.out.write(time.strftime("%m-%d %H:%M:%S "))
                self.out.write(part)
                self.line_start = part.endswith("\n")
        return len(text)

    def __getattr__(self, name):
        return getattr(self.out, name)


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)   # kill -USR1 PID: every thread's stack to stderr (the log)
    parser = argparse.ArgumentParser(description="Run SlackFish as a Lichess bot.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upgrade", help="turn the token's account into a BOT account (permanent; no games played yet)")
    p = sub.add_parser("play", help="go online: accept challenges, optionally play the Lichess AI")
    p.add_argument("--bot", type=parse_bot, default=("t3-sg64-muon-lichessgames3m-16.0m", 3),
                   help="NAME[:DEPTH]; DEPTH is when a pondered reply is played at once (default t3-sg64-muon-lichessgames3m-16.0m:3)")
    p.add_argument("--book", nargs="?", const="gm2001.bin", help="opening book in Models/Books (default with no name: gm2001.bin); NAME,NAME,... layers books, the first with a move for the position plays it")
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
                        f"(default: {MOVES_LEFT_BULLET} for bullet clocks, else {MOVES_LEFT})")
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
    p.add_argument("--color", choices=["white", "black", "random"], default="random",
                   help="with --challenge: the colour we ask for (default random); the opponent may decline it")
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
    sys.stdout = Timestamped(sys.stdout)
    runner = BotRunner(li, args)
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        runner.report()


if __name__ == "__main__":
    main()
