"""
Play against a SlackFish bot in the browser, or watch two bots play each other.

    uv run python Engine/gui.py                         # opens http://localhost:8000
    uv run python Engine/gui.py --bot t1-alpha:2 --color black
    uv run python Engine/gui.py --color watch --bot t2-w-6.41m:3 --bot2 t1-3.25m:3 --delay 1
    uv run python Engine/gui.py --temperature 50         # vary the bot's moves
    uv run python Engine/gui.py --book                   # open from Models/Books/gm2001.bin
    uv run python Engine/gui.py --quiescence 0          # turn off quiescence search
    uv run python Engine/gui.py --device cpu --port 8080 --no-browser

The board is Chessground (lichess's board UI, loaded from jsDelivr). Rules,
legal moves and the bots all run here in Python; the page just renders the
state and sends moves back. Everything (mode, each bot's model, depth,
quiescence and temperature, the clock, the watch delay) can also be set in the page.

Watch mode (--color watch): two bots play each other, waiting at least --delay
seconds per move so the game can be followed; the page can pause and resume it.

By default the bot always plays its best move, so the same position always gets
the same reply. --temperature (or the page's Temperature box), in centipawns,
makes it sample moves instead: at 50, a move 50 cp worse is e^-1 = 0.37x as likely.

Opening book (--book, or the page's Opening book menu): bots play moves from a
Polyglot .bin book in Models/Books, sampled by how often each was played, until
the game leaves the book or reaches --book-plies plies, then search as usual.

Optional clock for the human player (--clock 5+3, or set in the page): it runs
only on the human's turn and starts after their first move (as on lichess, so
loading the page doesn't cost time), the bot thinks for free, and running out loses unless
the bot has too little material to mate (then it's a draw). The server keeps the
time, so every viewer sees the same clock.

There is one shared game: every open page (including remote viewers through a
tunnel such as `ngrok http 8000`) sees the same board, live, and anyone with the
link can make moves, undo or start a new game.
"""
import argparse
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import chess.pgn

from bots import BOTS, ROOT, OpeningBook, load_bot
from export import game_gif, game_png
from play import game_over

HTML_PATH = os.path.join(os.path.dirname(__file__), "gui.html")
BOOKS_DIR = os.path.join(ROOT, "Models", "Books")


def available_bots():
    return [name for name, spec in BOTS.items() if os.path.exists(os.path.join(ROOT, spec.weights))]


def available_books():
    return sorted(f for f in os.listdir(BOOKS_DIR) if f.endswith(".bin")) if os.path.isdir(BOOKS_DIR) else []


class Game:
    """One shared game. Every viewer sees and controls the same board.

    All methods except _bot_worker are called with self.lock held. Each change
    bumps self.version and wakes the /api/events streams so every open page
    updates. The bot searches on a background thread, so the board stays
    responsive (and undo / new game still work) while it thinks.
    """

    def __init__(self, white, black, device=None, fen=None, clock=None, delay=0.0,
                 book=None, book_plies=16):
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.version = 0
        self.thinking = False
        self.device = device
        # Human clock: clock = (base seconds, increment seconds) or None for no clock
        self.clock = None
        self.time_left = None     # seconds, as of turn_start
        self.turn_start = None    # time.monotonic() when the running clock was started, else None
        self.flagged = False      # the human ran out of time
        self.clock_token = 0      # bumped on every stop, so a stale flag timer does nothing
        self.paused = False       # watch mode: don't start the next bot move
        self.book_plies = book_plies
        self.books = {}           # file name -> OpeningBook, opened on first use
        with self.lock:
            self.new(white, black, fen, clock, delay, book)

    def _changed(self):
        self.version += 1
        self.changed.notify_all()

    def new(self, white, black, fen=None, clock=None, delay=0.0, book=None):
        """white / black: a bot settings dict (see parse_bot) or None for the human.
        At most one side can be human; with none, the bots play each other.
        book: a file name from available_books() for the bots to open from, or None."""
        if white is None and black is None:
            raise ValueError("at least one side must be a bot")
        if book is not None and book not in available_books():
            raise ValueError(f"unknown opening book {book!r}")
        if book is not None and book not in self.books:
            self.books[book] = OpeningBook(os.path.join(BOOKS_DIR, book), max_plies=self.book_plies)
        self.book = book
        self.specs = {chess.WHITE: white, chess.BLACK: black}
        self.players = {color: None if spec is None else load_bot(
                            spec["name"], depth=spec["depth"], device=self.device,
                            temperature=spec["temperature"], quiescence_depth=spec["quiescence"],
                            book=self.books.get(book))
                        for color, spec in self.specs.items()}
        self.human_color = (chess.WHITE if white is None else chess.BLACK if black is None else None)
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_times = []                  # seconds spent on each move in move_stack
        self.turn_began = time.monotonic()    # when the side to move started its turn
        self.eval = None      # White-perspective score from the last bot search
        self.eval_by = None   # which bot made it
        self.eval_book = False   # the last bot move came from the opening book
        self.delay = delay if self.human_color is None else 0.0
        self.paused = False
        self._stop_clock()
        self.clock = clock if self.human_color is not None else None
        self.time_left = self.clock[0] if self.clock else None
        self.flagged = False
        self._changed()
        self._start_clock()
        self._start_bot_if_its_turn()

    # --- human clock -------------------------------------------------------

    def _over(self):
        return self.flagged or game_over(self.board)

    def _human_has_moved(self):
        plies = len(self.board.move_stack)
        human_first = self.board.root().turn == self.human_color
        return (plies + 1) // 2 > 0 if human_first else plies // 2 > 0

    def _start_clock(self):
        """Start the human's clock if it's their turn, after their first move, and
        it isn't already running."""
        if (self.clock is None or self.turn_start is not None or self._over()
                or self.board.turn != self.human_color or not self._human_has_moved()):
            return
        self.turn_start = time.monotonic()
        timer = threading.Timer(max(0.0, self.time_left) + 0.05, self._flag, args=(self.clock_token,))
        timer.daemon = True
        timer.start()

    def _stop_clock(self):
        """Stop the running clock and charge the time used to the human."""
        if self.turn_start is not None:
            self.time_left -= time.monotonic() - self.turn_start
            self.turn_start = None
        self.clock_token += 1

    def _clock_left(self):
        if self.time_left is None:
            return None
        used = time.monotonic() - self.turn_start if self.turn_start is not None else 0.0
        return max(0.0, self.time_left - used)

    def _flag(self, token):
        with self.lock:
            if token != self.clock_token or self._clock_left() > 0:
                return
            self._stop_clock()
            self.time_left = 0.0
            self.flagged = True
            self._changed()

    def _result(self):
        """(result, reason) if the game is over, else (None, None)."""
        if self.flagged:
            if self.board.has_insufficient_material(not self.human_color):
                return "1/2-1/2", "timeout vs insufficient material"
            return ("0-1" if self.human_color == chess.WHITE else "1-0"), "timeout"
        if game_over(self.board):
            outcome = self.board.outcome(claim_draw=True)
            return outcome.result(), outcome.termination.name.replace("_", " ").lower()
        return None, None

    def _start_bot_if_its_turn(self):
        bot = self.players[self.board.turn]
        if self.thinking or self._over() or bot is None or self.paused:
            return
        self.thinking = True
        self._changed()
        threading.Thread(target=self._bot_worker, args=(bot, self.board.copy(), self.version, self.delay),
                         daemon=True).start()

    def _bot_worker(self, bot, board, started_at, delay):
        start = time.monotonic()
        try:
            move, score = bot.search(board)   # slow part runs without the lock
        except Exception as e:
            print(f"bot search failed: {e}")
            move = None
        thought = time.monotonic() - start   # thinking time only, not the watch delay below
        time.sleep(max(0.0, delay - (time.monotonic() - start)))   # watch mode: let viewers follow
        with self.lock:
            self.thinking = False
            # Only play the move if nobody undid / restarted while the bot was thinking
            if move is not None and self.version == started_at:
                self._push(move, thought)
                self.eval_by, self.eval_book = bot.name, score is None
                if score is not None:   # a book move has no score: keep showing the last one
                    self.eval = score
                self._start_clock()
            self._changed()
            self._start_bot_if_its_turn()   # e.g. a new game started with the bot to move

    def state(self):
        b = self.board
        over = self._over()
        result, reason = self._result()
        dests = {}
        if not over and b.turn == self.human_color:
            for m in b.legal_moves:
                dests.setdefault(chess.square_name(m.from_square), []).append(chess.square_name(m.to_square))
        last = b.move_stack[-1] if b.move_stack else None

        history = []
        replay = b.root()
        for m in b.move_stack:
            history.append(replay.san(m))
            replay.push(m)
        if replay.move_stack and b.root().turn == chess.BLACK:
            history.insert(0, "…")   # keep White/Black columns aligned
        times = ([None] if len(history) > len(self.move_times) else []) + [round(t, 2) for t in self.move_times]

        return {
            "fen": b.fen(),
            "turn": "white" if b.turn == chess.WHITE else "black",
            "humanColor": None if self.human_color is None else "white" if self.human_color == chess.WHITE else "black",
            "white": self.specs[chess.WHITE],   # None = human
            "black": self.specs[chess.BLACK],
            "dests": dests,
            "lastMove": [chess.square_name(last.from_square), chess.square_name(last.to_square)] if last else None,
            "check": b.is_check(),
            "history": history,
            "times": times,   # seconds per move, aligned with history (None for the "…" filler)
            "gameOver": over,
            "result": result,
            "reason": reason,
            "flagged": self.flagged,
            "clock": None if self.clock is None else {
                "base": self.clock[0], "increment": self.clock[1],
                "left": self._clock_left(), "running": self.turn_start is not None,
            },
            "eval": self.eval,
            "evalBy": self.eval_by,
            "evalBook": self.eval_book,
            "book": self.book,
            "books": available_books(),
            "bots": available_bots(),
            "thinking": self.thinking,
            "thinkingBot": (self.players[b.turn].name if self.thinking and self.players[b.turn] else None),
            "paused": self.paused,
            "delay": self.delay,
            "version": self.version,
        }

    def human_move(self, uci: str):
        move = chess.Move.from_uci(uci)
        if self._over() or self.board.turn != self.human_color or move not in self.board.legal_moves:
            raise ValueError(f"illegal move {uci}")
        if self.clock is not None:
            self._stop_clock()
            if self.time_left <= 0:   # the move arrived after the flag fell
                self.time_left, self.flagged = 0.0, True
                self._changed()
                raise ValueError("out of time")
            self.time_left += self.clock[1]
        self._push(move, time.monotonic() - self.turn_began)
        self._changed()
        self._start_bot_if_its_turn()

    def undo(self):
        # Take back to the last position where it was the human's turn. Time
        # already used stays used, and there's no undo after losing on time.
        if self.flagged:
            raise ValueError("can't undo after running out of time")
        if self.human_color is None:
            raise ValueError("no undo when watching; start a new game instead")
        self._stop_clock()
        if self.board.move_stack:
            self.board.pop()
        while self.board.move_stack and self.board.turn != self.human_color:
            self.board.pop()
        del self.move_times[len(self.board.move_stack):]
        self.turn_began = time.monotonic()
        self.eval = None
        self.eval_book = False
        self._changed()
        self._start_clock()
        self._start_bot_if_its_turn()   # undoing to the start as Black: the bot moves again

    def _push(self, move, seconds):
        self.board.push(move)
        self.move_times.append(seconds)
        self.turn_began = time.monotonic()

    def set_paused(self, paused: bool):
        self.paused = bool(paused)
        self._changed()
        self._start_bot_if_its_turn()

    def player_names(self):
        return tuple("Human" if self.players[c] is None else str(self.players[c])
                     for c in (chess.WHITE, chess.BLACK))

    def export(self, fmt):
        white, black = self.player_names()
        render = game_gif if fmt == "gif" else game_png
        return render(self.board.copy(), white=white, black=black, flipped=self.human_color == chess.BLACK)

    def pgn(self):
        game = chess.pgn.Game.from_board(self.board)
        game.headers["White"], game.headers["Black"] = self.player_names()
        result, reason = self._result()
        if result:
            game.headers["Result"] = result
            if self.flagged:
                game.headers["Termination"] = "time forfeit"
        if self.clock:
            game.headers["TimeControl"] = f"{self.clock[0]:g}+{self.clock[1]:g} (human only)"
        return str(game)


def parse_clock(spec):
    """'5+3' (minutes + increment seconds) -> (300.0, 3.0); None / '' / 'none' -> None."""
    if not spec or str(spec).lower() == "none":
        return None
    minutes, _, increment = str(spec).partition("+")
    base, inc = float(minutes) * 60, float(increment or 0)
    if not (0 < base <= 3 * 3600 and 0 <= inc <= 300):
        raise ValueError("clock must be 0-180 minutes + 0-300 seconds")
    return base, inc


def parse_bot(spec, default_depth=2, default_quiescence=6, default_temperature=0.0):
    """Bot settings from the page (a dict) or the command line ('NAME[:DEPTH]'),
    validated so a remote viewer can't start an unbounded search."""
    if isinstance(spec, str):
        name, _, depth = spec.partition(":")
        spec = {"name": name, "depth": depth or default_depth}
    name = spec["name"]
    if name not in BOTS:
        raise ValueError(f"unknown bot {name!r}")
    depth = int(spec.get("depth") or default_depth)
    quiescence = int(spec.get("quiescence", default_quiescence))
    if not 1 <= depth <= 5:
        raise ValueError("depth must be 1-5")
    if not 0 <= quiescence <= 12:
        raise ValueError("quiescence must be 0-12 plies")
    return {"name": name, "depth": depth, "quiescence": quiescence,
            "temperature": parse_temperature(spec.get("temperature", default_temperature))}


def parse_delay(value):
    delay = float(value or 0)
    if not 0 <= delay <= 10:
        raise ValueError("delay must be 0-10 seconds")
    return delay


def parse_temperature(value):
    """Temperature in centipawns: None / '' -> 0 (always the best move)."""
    temperature = float(value or 0)
    if not 0 <= temperature <= 1000:
        raise ValueError("temperature must be 0-1000 centipawns")
    return temperature


def make_handler(game: Game):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, content_type="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                with open(HTML_PATH, "rb") as f:
                    self.send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                with game.lock:
                    self.send(200, json.dumps(game.state()))
            elif self.path == "/api/pgn":
                with game.lock:
                    self.send(200, game.pgn(), "text/plain; charset=utf-8")
            elif self.path in ("/api/export.png", "/api/export.gif"):
                fmt = self.path.rsplit(".", 1)[1]
                with game.lock:
                    data = game.export(fmt)
                self.send(200, data, f"image/{fmt}")
            elif self.path == "/api/events":
                self.stream_events()
            else:
                self.send(404, "not found", "text/plain")

        def stream_events(self):
            """Server-Sent Events: push the state to this viewer whenever it changes.

            One long-lived request per open page, so remote viewers (e.g. through
            ngrok) stay in sync without polling. A comment line every 15s keeps
            proxies from closing an idle connection; EventSource reconnects on its own.
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with game.changed:
                        if game.version == seen:
                            game.changed.wait(timeout=15)
                        payload = None
                        if game.version != seen:
                            seen = game.version
                            payload = json.dumps(game.state())
                    self.wfile.write(f"data: {payload}\n\n".encode() if payload else b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass   # viewer closed the page

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                with game.lock:
                    if self.path == "/api/move":
                        game.human_move(body["uci"])
                    elif self.path == "/api/undo":
                        game.undo()
                    elif self.path == "/api/new":
                        white = None if body.get("white") is None else parse_bot(body["white"])
                        black = None if body.get("black") is None else parse_bot(body["black"])
                        game.new(white, black, body.get("fen"), parse_clock(body.get("clock")),
                                 parse_delay(body.get("delay")), body.get("book") or None)
                    elif self.path == "/api/pause":
                        game.set_paused(body.get("paused", True))
                    else:
                        return self.send(404, json.dumps({"error": "not found"}))
                    self.send(200, json.dumps(game.state()))
            except Exception as e:
                self.send(400, json.dumps({"error": str(e)}))

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Play SlackFish in the browser.")
    parser.add_argument("--bot", default="t2-w-6.41m:2", help="bot name with optional :DEPTH")
    parser.add_argument("--color", choices=["white", "black", "watch"], default="white",
                        help="your colour, or watch: --bot (White) plays --bot2 (Black)")
    parser.add_argument("--bot2", help="watch mode: Black's bot as NAME[:DEPTH] (default: same as --bot)")
    parser.add_argument("--quiescence", type=int, default=6,
                        help="quiescence search plies past the depth limit for the bots (0 = off, default 6)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="watch mode: minimum seconds per move so the game can be followed (default 1)")
    parser.add_argument("--fen", help="start from this position")
    parser.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--clock", help="your clock as MINUTES+INCREMENT, e.g. 5+3 (default: no clock)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sample the bot's moves with this temperature in centipawns (default: 0, always the best move)")
    parser.add_argument("--book", nargs="?", const="gm2001.bin",
                        help="opening book file in Models/Books for the bots (default with no name: gm2001.bin)")
    parser.add_argument("--book-plies", type=int, default=16,
                        help="leave the book after this many plies at most (default 16)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    bot = lambda spec: parse_bot(spec, default_quiescence=args.quiescence, default_temperature=args.temperature)
    if args.color == "watch":
        white, black = bot(args.bot), bot(args.bot2 or args.bot)
    else:
        white, black = (None, bot(args.bot)) if args.color == "white" else (bot(args.bot), None)
    game = Game(white, black, args.device, args.fen, parse_clock(args.clock), parse_delay(args.delay),
                args.book, args.book_plies)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(game))
    url = f"http://localhost:{args.port}"
    print(f"SlackFish GUI running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
