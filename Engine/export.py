"""
Export a finished game as a PNG (final position) or an animated GIF (every move).

    from export import game_png, game_gif
    game_png(board, "final.png")
    game_gif(board, "game.gif", white="Human", black="t2-6.41m", flipped=True)

Boards are drawn with chess.svg (python-chess) and rasterised with cairosvg.
Each image has a caption strip under the board with the last move and, on the
final frame, the result.
"""
import io

import cairosvg
import chess
import chess.svg
from PIL import Image, ImageDraw, ImageFont

CAPTION_HEIGHT = 0.11   # fraction of the board size


def _result_text(board: chess.Board) -> str | None:
    from play import game_over   # shared draw rules
    if not game_over(board):
        return None
    outcome = board.outcome(claim_draw=True)
    return f"{outcome.result()}  ({outcome.termination.name.replace('_', ' ').lower()})"


def _frame(board: chess.Board, size: int, flipped: bool, caption: str) -> Image.Image:
    last = board.peek() if board.move_stack else None
    check = board.king(board.turn) if board.is_check() else None
    svg = chess.svg.board(board, size=size, flipped=flipped, lastmove=last, check=check)
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    board_img = Image.open(io.BytesIO(png)).convert("RGB")

    strip = int(size * CAPTION_HEIGHT)
    img = Image.new("RGB", (size, size + strip), (38, 36, 33))
    img.paste(board_img, (0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, int(strip * 0.45)))
    draw.text((size // 2, size + strip // 2), caption, fill=(236, 231, 223), font=font, anchor="mm")
    return img


def _caption(board: chess.Board, san: str | None, white: str, black: str) -> str:
    if san is None:
        return f"{white} (White) vs {black} (Black)"
    # board has already played the move: if it's White's turn now, Black just moved
    moved_black = board.turn == chess.WHITE
    number = board.fullmove_number - 1 if moved_black else board.fullmove_number
    return f"{number}{'...' if moved_black else '.'} {san}"


def game_png(board: chess.Board, path=None, size: int = 640, flipped: bool = False,
             white: str = "White", black: str = "Black") -> bytes:
    """Final position with the last move highlighted. Returns PNG bytes; also writes path if given."""
    caption = _result_text(board)
    if caption is None:
        caption = f"{white} vs {black}"
    buf = io.BytesIO()
    _frame(board, size, flipped, caption).save(buf, format="PNG")
    data = buf.getvalue()
    if path:
        with open(path, "wb") as f:
            f.write(data)
    return data


def game_gif(board: chess.Board, path=None, size: int = 480, flipped: bool = False,
             white: str = "White", black: str = "Black",
             move_ms: int = 700, end_ms: int = 4000) -> bytes:
    """Animated replay from the starting position to the current one. Returns GIF bytes."""
    replay = board.root()
    frames = [_frame(replay, size, flipped, _caption(replay, None, white, black))]
    for move in board.move_stack:
        san = replay.san(move)
        replay.push(move)
        frames.append(_frame(replay, size, flipped, _caption(replay, san, white, black)))

    result = _result_text(board)
    if result:   # hold on the final position with the result shown
        frames.append(_frame(replay, size, flipped, result))

    durations = [1500] + [move_ms] * (len(frames) - 2) + [end_ms] if len(frames) > 1 else [end_ms]
    palette_frames = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=64) for f in frames]
    buf = io.BytesIO()
    palette_frames[0].save(buf, format="GIF", save_all=True, append_images=palette_frames[1:],
                           duration=durations, loop=0, optimize=True)
    data = buf.getvalue()
    if path:
        with open(path, "wb") as f:
            f.write(data)
    return data
