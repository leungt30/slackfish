# SlackFish

SlackFish is a lightweight chess engine that uses a neural network to approximate Stockfish position evaluations. It converts chess positions from FEN into bitboards and engineered chess features, predicts a centipawn evaluation, and searches over those evaluations to choose moves.

The project was built under limited compute constraints, so the focus is on combining learned board representations with explicit chess knowledge such as material balance, mobility, castling rights, piece-square tables, hanging pieces, pawn structure, pins, forks, and center control.

## Quick Start

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies into .venv
uv sync

# 3. Play against the engine in the browser (default bot: t2-w-lichess-6.41m at depth 3)
uv run python Engine/gui.py
```

## Current Status

| Model | Code | Weights | Playable |
|---|---|---|---|
| **v1** | `Models/Training/Models.py` | `Models/Weights/SlackFishCNN_V1_499` | Yes |
| **v2** | `Models/Training/Models.py` | `Models/Weights/SlackFishCNN_V2_449` | Yes |
| v3, v4 | `Models/Training/Models.py` | none | No (not registered in `Engine/bots.py`) |
| **v5** | `Models/Training/Models.py` | **missing** (`SlackFishCNN_V5_79` was never committed) | No, needs retraining |
| **t1-3.25m** | `NoFeat/model.py` | `Models/Weights/SlackFishV-T1-3.25M/` | Yes: SlackFishV-T1, 3.25M params, all 14M positions, 10 epochs, val MAE 143 cp |
| **t1-alpha** | `NoFeat/model.py` | `Models/Weights/SlackFishV-T1-alpha/` | Yes: 2M random positions, 3 epochs, val MAE 194 cp; 5.5/6 vs v2 |
| **t2-6.41m** | `NoFeat/model.py` (`ChessTransformerV2`) | `Models/Weights/SlackFishV-T2-6.41M/` | Yes: SlackFishV-T2, 6.41M params, games + random, colour-swap augmentation, 6 epochs, val MAE 155 cp (mirrored 155) |
| **t2-w-6.41m** | `NoFeat/model.py` (`ChessTransformerV2`) | `Models/Weights/SlackFishV-T2-W-6.41M/` | Yes: T2 with a win-probability target, games + random, 8 epochs, val MAE games 128 / random 167 / tactics 630 cp |
| **t2-w-all-6.41m** | `NoFeat/model.py` (`ChessTransformerV2`) | `Models/Weights/SlackFishV-T2-W-All-6.41M/` | Yes: as t2-w, plus tactics, 8 epochs, val MAE games 142 / random 171 / tactics 356 cp |
| **t2-w-lichess-6.41m** (default) | `NoFeat/model.py` (`ChessTransformerV2`) | `Models/Weights/SlackFishV-T2-W-Lichess-6.41M/` | **Yes, strongest:** as t2-w, plus tactics and 393M Lichess positions, 1 epoch, val MAE games 114 / random 155 / tactics 293 cp. Scored 126/128 against t2-w; about 2255 Elo on Stockfish's scale at depth 3 |

Each NoFeat folder holds `model.safetensors` + `config.json` (the inference export, see [Sharing models](#sharing-models)) and/or the training checkpoint `best.pt`; the bots use the export when it's there.

V5 is the latest model described in `REPORT.md`. Its original checkpoint is lost, so it has to be retrained (see [Retraining V5](#retraining-v5)). Once a checkpoint exists at `Models/Weights/SlackFishCNN_V5_79`, or is passed with `--white-weights` / `--black-weights`, it plays with no code changes.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and declared in `pyproject.toml`, with exact versions locked in `uv.lock`.

```bash
uv sync                       # create .venv and install everything
uv run python <script>.py     # run any script inside the environment
source .venv/bin/activate     # or activate it and use python directly
uv add <package>              # add a new dependency
```

Dependencies: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `wandb`, `tqdm`, `python-chess`.

Notes:

- Python 3.12+ is required.
- On Linux, `torch` from PyPI ships with CUDA. The engine uses the GPU if `torch.cuda.is_available()`, and falls back to CPU otherwise.
- The saved scalers were pickled with scikit-learn 1.6.1. Newer versions load them with an `InconsistentVersionWarning`, which is safe to ignore for `StandardScaler`.

## Playing Games

### In the browser (recommended for human vs bot)

```bash
uv run python Engine/gui.py                              # opens http://localhost:8000, vs t2-w-lichess-6.41m:3
uv run python Engine/gui.py --bot t2-w-6.41m:2 --color black
uv run python Engine/gui.py --temperature 50              # sample moves instead of always the best (centipawns)
uv run python Engine/gui.py --book                       # bots open from the gm2001 book (also a menu in the page)
uv run python Engine/gui.py --device cpu                 # keep the GPU free, e.g. while training
uv run python Engine/gui.py --no-ponder                # don't let the bot think on your time
uv run python Engine/gui.py --fen "<fen>" --port 8080 --no-browser
```

The board is [Chessground](https://github.com/lichess-org/chessground), lichess's board UI, loaded from jsDelivr (needs internet; GPL-3.0). You can drag or click pieces, and it shows legal-move dots, last move and check highlights, and a promotion picker. The side panel picks the bot (only bots whose weights exist are listed), the depth and your colour, and has an evaluation bar from the bot's last search, a move list, Undo, Flip and PGN download. When the game ends, **Save GIF** (animated replay of every move) and **Save PNG** (final position with the result) appear. Python runs the rules (python-chess) and the bot; `Engine/gui.html` only displays the state.

While it's your turn the bot ponders: it searches your likely replies in the background, best-looking first. If you play one it has finished, it replies instantly; otherwise it searches as usual, starting from the positions it already scored. Pondering never changes which move the bot picks, only how fast.

**Sharing a game (e.g. with ngrok).** There is one shared game per server. Every open page sees the same board and updates live: the server pushes changes over Server-Sent Events (`/api/events`), so there's no polling and a viewer costs one long-lived connection. The bot runs on the server, so everyone sees "thinking…" and its reply. To let others watch or play remotely:

```bash
uv run python Engine/gui.py --port 8000 --no-browser
ngrok http 8000          # share the https://….ngrok-free.app URL
```

Anyone with the link can move, undo or start a new game; there are no accounts or spectator-only mode.

### In the terminal

`Engine/play.py` runs bot-vs-bot, human-vs-bot, or human-vs-human games. Each player is `human` or `BOT[:DEPTH]`.

```bash
# Human (white) vs v2 searching 2 plies
uv run python Engine/play.py --white human --black v2:2

# Bot vs bot, different models and depths, saved to a PGN
uv run python Engine/play.py --white v1:1 --black v2:3 --pgn v1_vs_v2.pgn

# Same depth for every bot, starting from a custom position
uv run python Engine/play.py --white v2 --black v1 --depth 2 --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

# Try a different checkpoint
uv run python Engine/play.py --white v5 --black v2 --white-weights path/to/SlackFishCNN_V5_119
```

| Option | Default | Meaning |
|---|---|---|
| `--white`, `--black` | `human`, `v2` | `human` or a bot name, optionally `:DEPTH` |
| `--depth` | `1` | Search depth for bots without an explicit `:DEPTH` |
| `--white-weights`, `--black-weights` | registry path | Override a bot's checkpoint file |
| `--fen` | start position | Start from a FEN |
| `--book [PATH]` | off | Play opening moves from a Polyglot `.bin` book (no path: `Models/Books/gm2001.bin`) |
| `--book-plies` | `16` | Leave the book after at most this many plies |
| `--pgn` | `game.pgn` | Where to save the game |
| `--max-moves` | none | Stop after N full moves |
| `--quiet` | off | Don't print the board in bot-only games |
| `--gif PATH` | none | Save an animated GIF replay of the game |
| `--png PATH` | none | Save a PNG of the final position |

**Human input:** moves in SAN (`e4`, `Nf3`, `exd5`, `e8=Q`) or UCI (`e2e4`, `e7e8q`). You can also type `moves` to list legal moves, `undo` to take back your last move and the bot's reply, or `quit` to stop and save the game.

### Opening book

With `--book`, bots play from a Polyglot opening book until the position leaves it, or until `--book-plies` plies have been played, and then they search as usual. Book moves are picked at random, weighted by how often each was played. Moves with under 2% of a position's weight are skipped. `--seed` makes the choice reproducible, so different seeds give different openings even at temperature 0.

`Models/Books/gm2001.bin` (built from GM games, from [gmcheems-org/free-opening-books](https://github.com/gmcheems-org/free-opening-books)) is included. Any other Polyglot book works: pass its path to `play.py`, or drop it into `Models/Books/` and it appears in the GUI's *Opening book* menu.

```bash
uv run python Engine/play.py --white t2-w-6.41m:2 --black t1-3.25m:2 --book --seed 7
```

### Search depth

- **Depth 1** scores every legal move with the model and plays the best one. This is how the original scripts played.
- **Depth 2+** runs minimax with alpha-beta pruning. Checkmate, stalemate, insufficient material, the 50-move rule and threefold repetition are scored exactly rather than by the model. Faster mates score higher.
- All positions at the last ply are evaluated in one batched forward pass per node.

Rough speed for v2 on an RTX 500 laptop GPU in the opening: depth 1 is instant, depth 2 about 0.1s, and depth 3 about 1.5–3s per move. Each extra ply multiplies the cost by roughly the number of legal moves. Feature engineering (`featureEng.py`) is the bottleneck, not the network.

### Using it from Python

```python
import sys; sys.path.insert(0, "Engine")
import chess
from bots import load_bot
from play import play_game, HumanPlayer

play_game(load_bot("v1", depth=1), load_bot("v2", depth=3), pgn_path="game.pgn")
play_game(HumanPlayer("Tim"), load_bot("v2", depth=2))

bot = load_bot("v2", depth=2)
move, score = bot.search(chess.Board())   # score is centipawns, White's perspective
```

Any object with a `choose_move(board) -> chess.Move` method can be a player.

### Adding a model

Register it in `BOTS` in `Engine/bots.py`:

```python
"v6": BotSpec(
    build_model=lambda: SlackFishCNN_V6(...),
    encode=encode_v6,                       # board -> (bitboard planes, scalar features)
    weights="Models/Weights/SlackFishCNN_V6_99",
    scaler_y="Models/Scalers/scaler_Y_v6.pkl",
    scaler_x=None,                          # set if features were scaled during training
),
```

The `encode` function must build features in exactly the same order as the training script.

## NoFeat: Models Without Feature Engineering

`NoFeat/` is the workspace for models that learn from the raw board only. It doesn't use `featureEng.py` or `preprocess.py`. It is not a bot itself: each trained model gets its own name and folder (SlackFishV-T1-alpha, SlackFishV-T1-3.25M, …). The current architecture is a transformer.

**Input: 66 tokens.** The board is encoded exactly as it stands, with no flipping, and everything is from White's point of view.

| Tokens | Vocabulary |
|---|---|
| 64 square tokens (a1 … h8) | 0 empty, 1–6 white P N B R Q K, 7–12 black P N B R Q K, 13 en-passant target square |
| 1 castling token | 4-bit value: white kingside, white queenside, black kingside, black queenside |
| 1 turn token | 0 white to move, 1 black to move |

**Model** (`NoFeat/model.py`, ~3.25M parameters): token embedding + learned positional embedding → 4 pre-LayerNorm transformer encoder layers (d_model 256, 8 heads, feed-forward 1024, GELU, dropout 0.1) → mean pool → MLP → 1 value.

**Target:** Stockfish centipawns from White's perspective (as in the CSVs), clipped to ±1500 and divided by 500, trained with MSE. Forced mates become ±(32000 − N) before clipping, so they end up at ±1500. Black's evaluation is the negative of White's; the search minimizes on Black's turn.

```bash
# 1. Encode all CSVs, each position tagged with its source (~40s for 16.6M positions) -> data/nofeat.npz (~1.1 GB)
uv run python -m NoFeat.prepare_data
uv run python -m NoFeat.prepare_data --limit 50000 --out data/nofeat_small.npz   # small test set
# ... plus the Lichess evaluation database (see Data): ~410M positions in ~20 min, written as a
# packed folder (squares two per byte, ~15 GB) so it fits in RAM for training
uv run python -m NoFeat.prepare_data --lichess /path/to/lichess_db_eval.jsonl.zst --out /data/slackfish/nofeat_all

# 2. Train a named model. --out is required; checkpoints go to <out>/{last,best}.pt
uv run python -m NoFeat.train --out Models/Weights/SlackFishV-T3 --epochs 10
uv run python -m NoFeat.train --out Models/Weights/SlackFishV-T3 --resume Models/Weights/SlackFishV-T3/last.pt --epochs 15
uv run python -m NoFeat.train --out /tmp/smoke --data data/nofeat_small.npz --epochs 2   # smoke test
# The t2-w-lichess-6.41m run: one pass over everything, validating a 200k subset every 10k steps
uv run python -m NoFeat.train --arch v2 --target winprob --sources games random tactics lichess \
    --epochs 1 --batch-size 512 --dropout 0 --val-frac 0.005 --val-every 10000 --save-every 5000 \
    --data /data/slackfish/nofeat_all --out Models/Weights/SlackFishV-T2-W-Lichess-6.41M

# 2b. Export for play and sharing: model.safetensors + config.json next to best.pt
uv run python -m NoFeat.export Models/Weights/SlackFishV-T3/best.pt
```

**3. Register it to play.** Add one line to `BOTS` in `Engine/bots.py`:

```python
"t3": NoFeatSpec("Models/Weights/SlackFishV-T3"),
```

Then it's available as `--black t3:2` in `play.py` and in the GUI's bot list.

**Choosing data:** `--sources` picks what to train on: `games` (chessData), `random` (random_evals) and `tactics` (tactic_evals). The default is `games random`, matching the original report. Validation is drawn from all three either way and reported per source; sources you didn't train on are marked `*`. Compare models using these per-source numbers: the overall val MSE changes meaning when the mix changes.

```text
epoch 2: train 0.8233  val MSE 0.7999  val MAE 315 cp  mirrored 309 cp  (27s)
  MAE by source: games 273  random 358  tactics* 831   (* = not trained on)
```

**Colour-swap augmentation:** half of the training positions (`--augment-prob 0.5`) are mirrored with colours, castling and turn swapped and the score negated (`NoFeat/augment.py`, equivalent to python-chess `board.mirror()`). `mirrored` in the log is validation MAE on the colour-swapped validation set. If it's far from the normal MAE, the model treats the two colours differently. SlackFishV-T1-3.25M, trained without augmentation, scores 136 cp as-is and 182 cp mirrored.

Useful training options: `--limit N`, `--batch-size` (default 1024), `--lr` (3e-4, warmup then cosine decay), `--d-model`, `--layers`, `--heads`, `--d-ff`, `--dropout`, `--clip`, `--scale`, `--precision` (`auto` picks bf16 on Ampere and newer, fp16 on older GPUs such as the RTX 20 series), `--val-every N` / `--val-subset` (a learning curve within long epochs, stored as `curve` in the checkpoint), `--wandb`, `--no-compile`. The checkpoint stores the model config and target scale, so `play.py` loads any size without code changes.

**Performance on an RTX 500 Ada laptop GPU (4 GB):** about 3,700 positions/s with `torch.compile`, so one epoch over all 14M positions takes about 62 minutes. Peak GPU memory is about 3 GB at batch size 1024. In play it's much faster than v1/v2, because there's no feature engineering: depth 2 takes about 0.2s per move.

**Smoke test results** (100k positions, 3 epochs): validation MAE fell from 331 to 286 cp. This only shows the pipeline works; see [Experiments](#experiments) for the full runs.

### Sharing models

Training checkpoints (`best.pt`, `last.pt`) are ~77 MB, two thirds of which is optimizer state, and loading them runs pickled Python (`torch.load(..., weights_only=False)`), so only load ones you trust. For playing and moving models between machines, export them:

```bash
uv run python -m NoFeat.export --all     # every Models/Weights/*/best.pt -> model.safetensors + config.json
```

`model.safetensors` holds only the weights (25.6 MB for the 6.41M models, 13 MB for 3.25M) and can't run code when loaded; `config.json` has the model class, size, score scale and a training summary. Evaluations are bit-identical to the checkpoint's. The bots load a folder's `model.safetensors` if present, else `best.pt`. If neither is there and `SLACKFISH_HF_REPO` names a Hugging Face model repo holding `<folder>/model.safetensors` + `config.json`, the model is downloaded on first use. Keep `last.pt` (gitignored) only if you want to resume training.

## Experiments

Two machines were used:

| | Laptop | Desktop |
|---|---|---|
| GPU | RTX 500 Ada (4 GB) | RTX 2080 Ti (11 GB, Turing, 250 W default / 280 W max power limit) |
| CPU / RAM | | Intel i7-7700 (4 cores, 8 threads) / 31 GB |
| Runs | T1-alpha, T1-3.25M, T2-6.41M, T2-W-6.41M, all search experiments | T2-W-All-6.41M, T2-W-Lichess-6.41M, the 128-game matches and the Stockfish ratings |

Three kinds of measurement:

- **Validation MAE:** mean absolute error against Stockfish on held-out positions, in centipawns clipped to ±1500, so it compares across targets.
- **Matches:** bot against bot at a fixed depth. The early ones used 8 named openings (Italian, Sicilian, QGD, King's Indian, French, Caro-Kann, English, Réti), each played once with each colour, so 16 games per pairing. Later ones (`Engine/match.py`) use 64 openings sampled 10 plies deep from the gm2001 book, each with both colours, so 128 games. Bots are deterministic, so the openings are the only source of variety. Games still going at move 150 count as draws. With 16 games the 95% interval on a score is about ±23 percentage points (roughly ±150 Elo); with 128 games about ±8 points (roughly ±50 Elo).
- **Ratings against Stockfish 19:** its `UCI_LimitStrength` / `UCI_Elo` levels serve as fixed anchors, and the bot's rating is fitted by maximum likelihood over all games. That scale is calibrated against engine rating lists, not human ratings, and it depends on how much time Stockfish gets per move.

### Training runs

All NoFeat transformers, 66-token input, AdamW at 3e-4 with warmup and cosine decay.

| Model | Architecture | Training data | Target | Colour-swap | Epochs | Time / epoch | Val MAE |
|---|---|---|---|---|---|---|---|
| T1-alpha | V1, 3.25M params | 2M-position random sample | cp (MSE) | no | 3 | ~9 min | 194 cp |
| T1-3.25M | V1, 3.25M | games + random (13.7M) | cp (MSE) | no | 10 | ~62 min | 143 cp (182 mirrored) |
| T2-6.41M | V2, 6.41M | games + random | cp (MSE) | 50% | 6 | ~2.6 h | 155 cp |
| **T2-W-6.41M** | V2, 6.41M | games + random | win probability (BCE) | 50% | 8 | ~2.6 h | **131 cp** |
| T2-W-All-6.41M | V2, 6.41M | games + random + tactics (16.3M) | win probability (BCE) | 50% | 8 | 52 min (2080 Ti, fp16) | 178 cp (all three sources) |
| **T2-W-Lichess-6.41M** | V2, 6.41M, no dropout | games + random + tactics + Lichess (407.7M) | win probability (BCE) | 50% | 1 | 19.3 h (2080 Ti, fp16) | **106 cp (all four sources), 106 mirrored** |

Val MAE by epoch:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| T1-3.25M (cp) | 189 | 179 | 169 | 161 | 155 | 151 | 149 | 144 | 143 |
| T2-6.41M (cp) | 188 | 177 | 167 | 162 | 157 | 155 | | | |
| T2-W-6.41M (win prob.) | 184 | 169 | 155 | 147 | 142 | 135 | 133 | 131 | |

T2-W-All and T2-W-Lichess validate on every source, including ones the others didn't train on, so compare models with the per-source MAE, not the overall number. T2-W and T2-W-All share their validation positions (same data file and seed). T2-W-Lichess has its own split of the bigger data set (0.5% held out), drawn from the same sources, so differences of a few cp against it are within noise:

| Val MAE by source | T2-W-6.41M (games + random) | T2-W-All-6.41M (+ tactics) | T2-W-Lichess-6.41M (+ tactics + Lichess) |
|---|---|---|---|
| games | 128 cp | 142 cp | **114 cp** |
| random | 167 cp | 171 cp | **155 cp** |
| tactics | 630 cp | 356 cp | **293 cp** |
| lichess | | | 105 cp |

T2-W-All by epoch (16.25M training positions, 31,747 steps of 512 per epoch):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| games | 183 | 176 | 171 | 162 | 155 | 148 | 145 | 142 |
| random | 362 | 323 | 292 | 260 | 218 | 189 | 178 | 171 |
| tactics | 509 | 459 | 430 | 412 | 391 | 377 | 360 | 356 |

Adding the tactics data roughly halved the error on tactical positions, at a cost of 14 cp on positions from real games. T2-W-All was trained on a different machine in float16 rather than bfloat16, so data is not the only difference. Its error was still falling at epoch 8.

**T2-W-Lichess learning curve.** One pass over 407.7M positions (796,386 steps of 512), with the learning rate warming up to 3e-4 and then following a cosine decay to 3e-5. Every 10,000 steps a fixed 200k-position validation subset was scored:

| Step | 10k | 50k | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 790k |
|---|---|---|---|---|---|---|---|---|---|---|
| Positions seen | 5M | 26M | 51M | 102M | 154M | 205M | 256M | 307M | 358M | 404M |
| Overall | 236 | 201 | 183 | 160 | 143 | 128 | 120 | 114 | 109 | 107 |
| games | 199 | 174 | 160 | 146 | 137 | 127 | 122 | 118 | 115 | 115 |
| random | 392 | 372 | 343 | 257 | 224 | 186 | 165 | 164 | 153 | 153 |
| tactics | 602 | 552 | 543 | 492 | 424 | 374 | 334 | 312 | 298 | 291 |

It passed T2-W's final error on game positions (128) after about 205M positions, about 1.9× the ~110M position-updates T2-W got from 8 epochs over 13.7M positions. Over the last 150k steps, with the learning rate near its minimum, the games error stayed at 115–118: the 6.41M model looks close to its capacity on this data.

**Training choices for the single-epoch run:** no dropout (every position is new, so there's nothing to memorize), a 0.5% validation split (2.05M positions; 2% would be 8M and slow to score), and a validation subset every 10k steps for a learning curve. All 409,798,830 positions are held in RAM by packing the 64 square tokens two per byte (4 bits each): about 38 bytes per position, 15 GB on disk, and about 25 GB of the desktop's 31 GB RAM in use while training.

**Win-probability target** (`--target winprob`): the label is `sigmoid(cp / 400)` and the model outputs a logit, trained with binary cross-entropy. In play, `logit × 400` is used as a centipawn-like score, so search, sampling and the eval bar work unchanged. Errors in balanced positions (0 vs +200 cp is 50% vs 62%) cost far more than errors in won ones (+1000 vs +1500 is 92% vs 98%), and forced mates need no clipping.

### Match results

| Match (depth 2, 16 games each) | Score | W / D / L | Rough Elo |
|---|---|---|---|
| T2-W-6.41M vs T2-6.41M | 10.5 / 16 | 9 / 1 / 4 (+2 unfinished) | +110 |
| T2-W-6.41M vs T1-3.25M | 11 / 16 | 10 / 1 / 4 (+1 unfinished) | +135 |

| T2-W-All-6.41M vs T2-W-6.41M | 7 / 16 | 4 / 6 / 6 | −44 |
| T2-W-All-6.41M vs T2-6.41M | 10.5 / 16 | 7 / 7 / 2 | +112 |
| T2-W-All-6.41M vs T1-3.25M | 10 / 16 | 6 / 8 / 2 | +89 |

| Match (depth 2, 128 games: 64 book openings × both colours) | Score | W / D / L | Rough Elo (95% interval) |
|---|---|---|---|
| T2-W-All-6.41M vs T2-W-6.41M | 52.5 / 128 (41%) | 32 / 41 / 55 | −63 (−115 to −14) |
| **T2-W-Lichess-6.41M vs T2-W-6.41M** | **126 / 128 (98%)** | **124 / 4 / 0** | **about +700** (too lopsided to pin down) |

Earlier, smaller checks of 2–4 games: T1-3.25M drew T1-alpha 1–1. T2-6.41M beat T1-3.25M 2–0, despite its higher validation error.

The 16-game T2-W-All vs T2-W result (−44) looked like a tie; the 128-game match showed T2-W-All is genuinely weaker. T2-W-Lichess won every decisive game against T2-W, with identical scores as White and Black (63/64 each). The two networks' raw evaluations suggest why: T2-W rates balanced positions as clearly better for White (+77 cp at the start, +140 in an Italian Game), while T2-W-Lichess gives +28 and +71, close to what strong engines say. At depth 2 the bot leans almost entirely on its evaluation, so a systematic bias gets exploited in every game.

### Strength against Stockfish

`Engine/match.py BOT stockfish:LEVELS`: 12 book openings × both colours × each level, Stockfish 19 on one thread.

| Bot | Stockfish time / move | Levels (games each) | Rating (Stockfish scale, 95% interval) |
|---|---|---|---|
| T2-W-6.41M, depth 2 | 0.3 s | 1320 (8 games) | 1 / 8: very roughly 1000 |
| T2-W-Lichess-6.41M, depth 2 | 0.5 s | 1320–2100 (24) | **1972 ± 84** |
| T2-W-Lichess-6.41M, depth 3 | 1 s | 1700–2500 (24) | **2255 ± 80** |

T2-W-Lichess at depth 3 against each level:

| Stockfish level | 1700 | 1900 | 2100 | 2300 | 2500 |
|---|---|---|---|---|---|
| Score / 24 | 19.5 | 20.5 | 13.5 | 12.5 | 10.5 |
| W / D / L | 19 / 1 / 4 | 19 / 3 / 2 | 10 / 7 / 7 | 11 / 3 / 10 | 9 / 3 / 12 |

At depth 2 it won all 24 games against level 1320 and 23 of 24 against 1500. It still scored 44% against the strongest level tested, so the top of its range isn't pinned down. Depth 3 adds roughly 300 Elo, though part of that gap is Stockfish getting 1 s instead of 0.5 s. A depth-3 move takes a median 0.73 s (1.7 s at most) on the desktop.

### Hardware and speed

| Experiment | Result |
|---|---|
| **bf16 vs fp16 training on the RTX 2080 Ti** | bf16 is emulated on Turing: 2.8 steps/s (batch 512) vs 9.8 steps/s for fp16 with loss scaling, 3.5× faster with the same losses. `--precision auto` now picks fp16 on pre-Ampere GPUs. An 8-epoch run went from an estimated 25 h to 7 h. |
| **Attention kernels** | Flash attention and cuDNN attention need Ampere or newer (FlashAttention-2 dropped Turing). The memory-efficient (CUTLASS) kernel built for sm75 is what PyTorch picks, and is 2.4× faster than the plain math fallback (9.0 vs 22.1 ms fwd+bwd per layer at batch 512). With 66 tokens attention is a small part of the work; most of it is the feed-forward layers. |
| **Power limit 250 W → 280 W** | Training was power-capped (248 W of 250, "SW Power Cap" active, 74 °C). At 280 W the clock rose from ~1710 to ~1770 MHz (+3.5%) but training only from ~11.4 to ~11.6 steps/s (+1–2%), at 78 °C. |
| **Whole-data packing** | 64 square tokens at 4 bits each instead of a byte: 409.8M positions in 15 GB instead of ~29 GB, unpacked on the GPU per batch at no measurable cost (11.5 steps/s, faster than the unpacked 16M-position run). |
| **Parallel games on one GPU** | Each game process uses ~400 MB of GPU memory but a full CPU core (the search is Python). 3 processes: ~40 s per depth-2 game, 4.5 games/min. 6 processes: ~57 s per game, 6.3 games/min. The 8-thread CPU, not the GPU, sets the limit at about 6–7. |
| **Pondering** (thinking on the opponent's time) | A predicted reply is answered in 0.001 s instead of 1.2 s (depth 3, same move and score as a fresh search). A wrong prediction aborts the background search within ~5 ms, and the search caches stay warm. |
| **Exported weights** | `model.safetensors` is 25.6 MB vs 77 MB for `best.pt` (optimizer state dropped), with bit-identical evaluations. |

### Search and inference

Same T2-W network on both sides unless noted. "Evals" means positions scored by the network.

| Experiment | Result |
|---|---|
| **Profiling** (one depth-3 search) | About 2/3 of search time is the network on the GPU. It costs ~0.4 ms per position whether batches are 30 or 900 positions, so the GPU is compute-bound, not latency-bound. |
| **float16 inference** (autocast) | 2.6× faster per batch (12.1 → 4.7 ms for 30 positions). Scores within 0.8 cp of float32 on average (5.9 cp max). bfloat16 was similar in speed but less accurate (3.2 cp mean). Whole searches got 2.5–2.8× faster. |
| **Evaluation cache** (position → score, per bot) | Within one search, transpositions save 19% of evals at depth 3. Root scores unchanged (≤ 0.0003 cp). |
| **Reusing caches across turns** | About 1% fewer evals (20 plies at depth 3: 102,208 vs 103,078) and no time saved. The new search's deepest plies are always new: the previous search only reached 1 ply below the new root. |
| **Sampling with temperature** | Exact scores for every root move cost 3.7× a normal depth-3 search. Pruning moves more than 4 × T worse than the best cut that to 32k evals instead of 103k. The move distribution barely changes: total variation ≤ 0.09, 0.01 on average. |
| **Transposition table + iterative deepening + network-score move ordering** | Alone, 7% fewer evals at depth 3 with identical results. The old search already pruned reasonably well. |
| **Quiescence search** (captures and check escapes, up to 6 plies) | Cost per move: depth 2 0.26 → 0.41 s, depth 3 0.76 → 1.31 s. Standing pat ends most capture lines at once. |
| New search vs old, equal depth (d2 vs d2) | 9.5 / 16 (6 / 7 / 3), about +65 Elo, at 1.5× the time per move |
| New search vs old one ply deeper (d2 vs d3) | 5 / 16 (2 / 6 / 8). One more full ply still wins, but the old search used 4× the time per move. |
| **New search vs old, equal time (d3 vs d3, 1.30 vs 1.27 s per move)** | **11 / 16 (8 / 6 / 2), about +135 Elo** |

### What we learned

1. **Better training targets and better search each helped more than a bigger model.** Switching to the win-probability target (+~120 Elo) and quiescence with move ordering (+~135 Elo at equal time) each cost nothing in model size.
2. **Doubling the model did almost nothing on this data.** T1 (3.25M) and T2 (6.41M) had the same validation error at every epoch. With ~14M positions for 6.4M parameters, data is the limit: T1's gap between training and validation loss grew from 0.006 to 0.027 by epoch 10. Scaling the model only makes sense alongside more data.
3. **Lower validation error doesn't guarantee stronger play.** T2 (155 cp) beat T1 (143 cp) in games. Match results decide, not MAE alone.
4. **Colour-swap augmentation fixes colour bias.** Without it, T1-3.25M scored 136 cp as-is but 182 cp on mirrored positions. With it, the two numbers agree within ~1 cp.
5. **Batching matters more than caching.** Evaluating all of a node's children in one forward pass is the main speed trick. Caches and the transposition table add only modest savings on top of it.
6. **Small samples mislead.** Single games between deterministic bots are repeats of each other, and even 16-game matches leave about ±150 Elo of uncertainty: T2-W-All looked even with T2-W over 16 games (−44) but was clearly weaker over 128 (−63, interval −115 to −14). Use varied openings, or the opening book, and as many games as time allows.
7. **More data beat everything else, by far.** Going from 16M to 409M positions (deeper Stockfish evaluations of positions from real play) took the same 6.41M model from T2-W's level to a 126–2 match score against it and about 2255 Elo on Stockfish's scale at depth 3, the largest single gain in the project. No change to the model, target or search came close.
8. **A small validation gain can be a large playing gain.** On game positions the error only went from 128 to 114 cp, yet the match was 126–2. The CSV labels are Stockfish 11 at depth 22, and MAE against them understates how much better the new evaluations are; systematic biases (like T2-W's +77 cp for White at the start) cost far more in play than random error does.
9. **Mixing in a narrow data source can hurt.** Adding 2.6M tactics positions to 14M others improved tactics but made the bot weaker (−63 Elo), since it took a sixth of training away from ordinary positions. Diluted inside 409M positions, the same tactics data no longer hurt.
10. **Check the hardware's number formats.** On Turing GPUs bf16 is emulated: fp16 with loss scaling trained 3.5× faster. Raising the power limit gave only 1–2%.
11. **With this much data, model size is likely the next limit.** The single-epoch curve flattened near 115 cp on game positions while the learning rate was low, unlike the 16M-position runs, where size made no difference. A larger model (e.g. width 384, 8 heads of 48) is the natural next experiment.

Ideas not yet tried: a larger model on the full data, more epochs over it, a steeper win-probability curve (`--wp-scale` ~200), true win/draw/loss labels, time management (the bot searches to a fixed depth, so it can lose on time in the GUI), and fewer, wider attention heads (16 heads of 16 dimensions are small for GPU kernels).

## Repository Layout

```text
.
├── Engine/
│   ├── bots.py              # Model registry, feature encoding, evaluator, minimax search
│   ├── play.py              # Game runner + CLI (bot/human vs bot/human)
│   ├── gui.py, gui.html     # Browser GUI for human vs bot (Chessground board)
│   ├── match.py             # Strength tests: bot vs bot, or vs Stockfish levels for an Elo estimate
│   ├── export.py            # Game -> animated GIF / final-position PNG
│   ├── GameLoop.py          # Legacy, broken (see below)
│   ├── BotVsBot.py          # Legacy, needs V5 weights
│   ├── startWithBoard.py    # Legacy, broken
│   └── testAllMovesFromFen  # Legacy, broken
├── NoFeat/
│   ├── encode.py            # FEN -> 64 square tokens + castling token + turn token
│   ├── model.py             # ChessTransformerV1, ChessTransformerV2
│   ├── augment.py           # Colour-swap augmentation
│   ├── prepare_data.py      # CSVs (+ Lichess) -> .npz or a packed folder
│   ├── train.py             # Training loop
│   ├── export.py            # Checkpoint -> model.safetensors + config.json
│   └── evaluator.py         # Load a model for play
├── Models/
│   ├── Scalers/             # scaler_X.pkl, scaler_Y.pkl (v1/v2), scaler_Y_9M.pkl (v5)
│   ├── Training/            # Model definitions and training scripts
│   ├── Weights/             # Model checkpoints and exports, one folder per NoFeat model
│   └── Books/               # Polyglot opening books (gm2001.bin)
├── data/                    # Raw CSVs, not committed (gitignored)
├── featureEng.py            # FEN-to-feature engineering
├── preprocess.py            # CSV -> preprocessed .npz
├── test.py                  # Inspect preprocessed data
├── pyproject.toml, uv.lock  # Dependencies
└── REPORT.md                # Original final report
```

## Model Architecture

SlackFish trains CNN regressors on positions labelled with Stockfish evaluations. It learns to evaluate positions rather than to predict moves.

| | v1 / v2 | v5 |
|---|---|---|
| Board input | 12 piece bitboards | 12 piece bitboards + 12 feature bitboards (24 × 8 × 8) |
| Scalar features | 11, standardized with `scaler_X.pkl` | 24, unscaled |
| Network | Conv layers → fully connected | 6 conv layers + 4 residual blocks → 7 fully connected layers, with the raw board also fed to the FC head |
| Output scaler | `scaler_Y.pkl` | `scaler_Y_9M.pkl` |
| Training data | — | ~9M positions |

Training uses MSE loss with Adam (`weight_decay=1e-4`). The model's output is inverse-transformed into centipawns from White's perspective.

### Features

**Bitboard channels:** one per piece type and color, isolated pawns, doubled pawns, rooks on the 7th rank, rooks on semi-open files, rooks on the same file, and hanging pieces (each feature has a white and a black channel).

**Scalar features (v5 order):** material count, side to move, en passant available, in check, castling rights (4), piece-square-table score, check available in one move (2), legal moves per side (2), forks (2), bishop activity (2), pinned pieces (2), hanging-piece value (2), center control (2), and win state.

**Scalar features (v1/v2 order):** material count, legal moves per side (2), side to move, en passant, in check, castling rights (4), and piece-square-table score.

## Data

Source: [Chess Evaluations on Kaggle](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations) (MIT license; originally from [r2dev2/ChessData](https://github.com/r2dev2/ChessData)). Downloading requires a Kaggle account.

| File | Rows | Contents | Used |
|---|---|---|---|
| `chessData.csv` | 12.96M | Positions from real games, Stockfish 11 depth 22 | Yes |
| `random_evals.csv` | 1.00M | Positions after random moves | Yes (in the original V5) |
| `tactic_evals.csv` | 2.63M | Puzzle positions + best move; median \|eval\| 450 cp, 10.7% forced mates | Optional: `--sources ... tactics` in NoFeat |

Each CSV has `FEN,Evaluation` columns. Evaluations are centipawns (`+56`, `-10`) or forced mates (`#+3`, `#-2`); `preprocess.py` converts a mate in N to ±(32000 − N) centipawns.

Put the files in `data/`:

```text
data/chessData.csv       # 12,958,035 positions
data/random_evals.csv    # 1,000,273 positions
data/tactic_evals.csv    # 2,628,219 positions
```

`data/` is gitignored. In git worktrees it can be a symlink to the main checkout's `data/`, so the large files live in one place.

### Lichess evaluation database

The [Lichess evaluation database](https://database.lichess.org/#evals) (`lichess_db_eval.jsonl.zst`, 22.4 GB compressed, CC0) has 416,442,401 positions analysed with Stockfish by Lichess users, one JSON line each: a FEN without move counters, and one or more evaluations, each with its search depth and principal variations (`cp` or `mate`, from White's perspective, best line first). `NoFeat.prepare_data --lichess` streams it through `zstd -dc`, takes each position's deepest evaluation, and keeps those searched to depth 20 or more (`--min-depth`): 393,212,303 positions (94.4%). The median depth is about 37, far deeper than the CSVs' depth 22. The positions are ones people chose to analyse, so they lean towards openings, sharp middlegames and endgames rather than random play; 12% of the combined set are forced mates.

## Retraining V5

**Status: in progress.** Both CSVs are downloaded; the pipeline below has not been run yet.

1. **Get the data:** `chessData.csv` and `random_evals.csv` into `data/` (see [Data](#data)).
2. **Preprocess:** `uv run python preprocess.py` computes every feature for every row and writes `preprocessed_chess_data.npz`. This is CPU-bound and takes hours for ~13M rows.
3. **Train:** `uv run python Models/Training/training_run_V5.py`. It logs metrics to Weights & Biases (`wandb login` first, or set `WANDB_MODE=offline`).
4. **Install the result:** copy the chosen checkpoint to `Models/Weights/` and the new target scaler to `Models/Scalers/`, then update the `v5` entry in `Engine/bots.py` if the file names differ.

Known problems to fix before a full run:

- **File name mismatch:** `preprocess.py` writes `preprocessed_chess_data.npz`, but `training_run_V5.py` reads `preprocessed_chess_data_dec_05_9M.npz`.
- **Output paths:** checkpoints are saved to `CNN/archive/SlackFishCNN_V5/`, outside `Models/Weights/`, and the directory must exist. The original V5 weights stayed in that folder and were never committed.
- **Target scaler not saved:** the `joblib.dump` calls are commented out. The new `scaler_Y` must be saved with the weights; the existing `scaler_Y_9M.pkl` belongs to the lost model.
- **Memory:** the script loads the whole dataset into RAM, and the default batch size (2048) is sized for a larger GPU. On a 4 GB GPU / 30 GB RAM machine, use a subset or a memory-mapped loader and a smaller batch size.
- **Hard-coded CUDA:** the resume path uses `map_location="cuda"`.

## Legacy Scripts (known issues)

The original scripts in `Engine/` and at the repo root predate `play.py` and are kept for reference. Use `Engine/play.py` instead.

| Script | Problem |
|---|---|
| `Engine/GameLoop.py` | Crashes: calls `featureEng.piece_mobility`, which was replaced by `legal_moves_per_side` |
| `Engine/startWithBoard.py` | Crashes: same `piece_mobility` problem |
| `Engine/testAllMovesFromFen` | Crashes: `piece_mobility`, imports `slackFishCNN` (wrong case), loads nonexistent `SlackFishCNN_499` |
| `Engine/BotVsBot.py` | Crashes: needs the missing V5 checkpoint |
| `Error Testing.py` | Crashes: needs the missing V5 checkpoint |

These scripts share some other problems:

- **Zeroed features:** v1/v2 features are normalized with `Scaler_X.fit_transform(...)` on a single position, which sets every engineered feature to 0. The correct call is `Scaler_X.transform(...)`, which `Engine/bots.py` uses. Because of this, v1 and v2 play differently in `play.py` than they did in the old scripts.
- **CUDA required:** `torch.load(..., map_location="cuda")` fails on machines without CUDA.
- **Run from repo root:** file paths are relative to the repo root.
- **Duplicate model code:** V1–V3 are defined in both `Models/Training/SlackFishCNN.py` and `Models/Training/Models.py`. The V1/V2 definitions are identical; `Engine/bots.py` uses `Models.py`.

## Evaluation

The report used an 80/20 train-test split with random shuffling each epoch. Cross-validation was not used because the dataset was already very large, with roughly 2 million samples in the test split.

The main metric was mean squared error against Stockfish evaluations. For qualitative evaluation, the engine's top-scoring moves were compared with Stockfish's top moves on selected positions. The original V5 scored 8/10 on a small tactical scenario set, compared with 3.5/10 for an older model.

## Known Limitations

- Endgame evaluation is less reliable.
- Checkmate sequences are hard for the evaluator alone. Searching at depth 2+ helps because mates are scored exactly.
- Move selection is deterministic, so bot-vs-bot games between the same bots always repeat.
- FEN inputs don't encode game history, so the model can't learn to avoid repetition. The search handles threefold repetition as a draw.
- v1 and v2 are weak in practice. They often play odd moves like `f6`, `h6` or `Rh7` in the opening.

Future improvements suggested in the report include endgame-specific features, a dedicated endgame model, stronger checkmate-oriented features, and adjusted mate-score scaling.

## References

The project was influenced by Stockfish-style evaluation, bitboard representations, piece-square tables, Maia Chess, DeepChess/AlphaZero-style hybrid approaches, and ResNet residual connections. See `REPORT.md` for the full discussion and source links.
