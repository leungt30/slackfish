"""
Play against a SlackFish bot in the browser.

    uv run python Engine/gui.py                         # opens http://localhost:8000
    uv run python Engine/gui.py --bot t1-alpha:2 --color black
    uv run python Engine/gui.py --device cpu --port 8080 --no-browser

The board is Chessground (lichess's board UI, loaded from jsDelivr). Rules,
legal moves and the bot all run here in Python; the page just renders the
state and sends moves back. Bot, depth and colour can also be changed in the page.
"""
import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import chess.pgn

from bots import BOTS, ROOT, load_bot
from export import game_gif, game_png
from play import game_over

HTML_PATH = os.path.join(os.path.dirname(__file__), "gui.html")


def available_bots():
    return [name for name, spec in BOTS.items() if os.path.exists(os.path.join(ROOT, spec.weights))]


class Game:
    def __init__(self, bot_name: str, depth: int, human_color: chess.Color, device=None, fen=None):
        self.lock = threading.Lock()
        self.device = device
        self.new(bot_name, depth, human_color, fen)

    def new(self, bot_name, depth, human_color, fen=None):
        self.bot = load_bot(bot_name, depth=depth, device=self.device)
        self.human_color = human_color
        self.board = chess.Board(fen) if fen else chess.Board()
        self.eval = None   # White-perspective score from the bot's last search

    def state(self):
        b = self.board
        over = game_over(b)
        outcome = b.outcome(claim_draw=True) if over else None
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

        return {
            "fen": b.fen(),
            "turn": "white" if b.turn == chess.WHITE else "black",
            "humanColor": "white" if self.human_color == chess.WHITE else "black",
            "dests": dests,
            "lastMove": [chess.square_name(last.from_square), chess.square_name(last.to_square)] if last else None,
            "check": b.is_check(),
            "history": history,
            "gameOver": over,
            "result": outcome.result() if outcome else None,
            "reason": outcome.termination.name.replace("_", " ").lower() if outcome else None,
            "bot": self.bot.name,
            "depth": self.bot.depth,
            "eval": self.eval,
            "bots": available_bots(),
        }

    def human_move(self, uci: str):
        move = chess.Move.from_uci(uci)
        if self.board.turn != self.human_color or move not in self.board.legal_moves:
            raise ValueError(f"illegal move {uci}")
        self.board.push(move)

    def bot_move(self):
        if game_over(self.board) or self.board.turn == self.human_color:
            return
        move, score = self.bot.search(self.board)
        self.eval = score
        self.board.push(move)

    def undo(self):
        # Take back to the last position where it was the human's turn
        if self.board.move_stack:
            self.board.pop()
        while self.board.move_stack and self.board.turn != self.human_color:
            self.board.pop()
        self.eval = None

    def players(self):
        human, bot = "Human", str(self.bot)
        return (human, bot) if self.human_color == chess.WHITE else (bot, human)

    def export(self, fmt):
        white, black = self.players()
        render = game_gif if fmt == "gif" else game_png
        return render(self.board.copy(), white=white, black=black, flipped=self.human_color == chess.BLACK)

    def pgn(self):
        game = chess.pgn.Game.from_board(self.board)
        game.headers["White"], game.headers["Black"] = self.players()
        if game_over(self.board):
            game.headers["Result"] = self.board.outcome(claim_draw=True).result()
        return str(game)


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
            else:
                self.send(404, "not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                with game.lock:
                    if self.path == "/api/move":
                        game.human_move(body["uci"])
                    elif self.path == "/api/bot":
                        game.bot_move()
                    elif self.path == "/api/undo":
                        game.undo()
                    elif self.path == "/api/new":
                        game.new(body["bot"], int(body["depth"]),
                                 chess.WHITE if body["color"] == "white" else chess.BLACK,
                                 body.get("fen"))
                    else:
                        return self.send(404, json.dumps({"error": "not found"}))
                    self.send(200, json.dumps(game.state()))
            except Exception as e:
                self.send(400, json.dumps({"error": str(e)}))

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Play SlackFish in the browser.")
    parser.add_argument("--bot", default="t1-3.25m:2", help="bot name with optional :DEPTH")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="your colour")
    parser.add_argument("--fen", help="start from this position")
    parser.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    name, _, depth = args.bot.partition(":")
    game = Game(name, int(depth or 1), chess.WHITE if args.color == "white" else chess.BLACK,
                args.device, args.fen)

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
