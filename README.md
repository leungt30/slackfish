# SlackFish

SlackFish is a lightweight chess engine that uses a neural network to approximate Stockfish position evaluations. It converts chess positions from FEN into bitboards and engineered chess features, predicts a centipawn evaluation, and searches over those evaluations to choose moves.

The project was built under limited compute constraints, so the focus is on combining learned board representations with explicit chess knowledge such as material balance, mobility, castling rights, piece-square tables, hanging pieces, pawn structure, pins, forks, and center control.

## Quick Start

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies into .venv
uv sync

# 3. Play against the engine
uv run python Engine/play.py --white human --black v2:2
```

## Current Status

| Model | Code | Weights | Playable |
|---|---|---|---|
| **v1** | `Models/Training/Models.py` | `Models/Weights/SlackFishCNN_V1_499` | Yes |
| **v2** | `Models/Training/Models.py` | `Models/Weights/SlackFishCNN_V2_449` | Yes |
| v3, v4 | `Models/Training/Models.py` | none | No (not registered in `Engine/bots.py`) |
| **v5** | `Models/Training/Models.py` | **missing** (`SlackFishCNN_V5_79` was never committed) | No, needs retraining |
| **t1-3.25m** | `NoFeat/model.py` | `Models/Weights/SlackFishV-T1-3.25M/best.pt` | Yes: SlackFishV-T1, 3.25M params, all 14M positions, 10 epochs, val MAE 143 cp |
| **t1-alpha** | `NoFeat/model.py` | `Models/Weights/SlackFishV-T1-alpha/best.pt` | Yes: 2M random positions, 3 epochs, val MAE 194 cp; 5.5/6 vs v2 |
| **t2-6.41m** | `NoFeat/model.py` (`ChessTransformerV2`) | `Models/Weights/SlackFishV-T2-6.41M/best.pt` | Yes: SlackFishV-T2, 6.41M params, games + random, colour-swap augmentation, 6 epochs, val MAE 155 cp (mirrored 155) |

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
uv run python Engine/gui.py                              # opens http://localhost:8000
uv run python Engine/gui.py --bot t1-alpha:3 --color black
uv run python Engine/gui.py --device cpu                 # keep the GPU free, e.g. while training
uv run python Engine/gui.py --fen "<fen>" --port 8080 --no-browser
```

The board is [Chessground](https://github.com/lichess-org/chessground), lichess's board UI, loaded from jsDelivr (needs internet; GPL-3.0). You can drag or click pieces, and it shows legal-move dots, last move and check highlights, and a promotion picker. The side panel picks the bot (only bots whose weights exist are listed), the depth and your colour, and has an evaluation bar from the bot's last search, a move list, Undo, Flip and PGN download. When the game ends, **Save GIF** (animated replay of every move) and **Save PNG** (final position with the result) appear. Python runs the rules (python-chess) and the bot; `Engine/gui.html` only displays the state.

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
| `--pgn` | `game.pgn` | Where to save the game |
| `--max-moves` | none | Stop after N full moves |
| `--quiet` | off | Don't print the board in bot-only games |
| `--gif PATH` | none | Save an animated GIF replay of the game |
| `--png PATH` | none | Save a PNG of the final position |

**Human input:** moves in SAN (`e4`, `Nf3`, `exd5`, `e8=Q`) or UCI (`e2e4`, `e7e8q`). You can also type `moves` to list legal moves, `undo` to take back your last move and the bot's reply, or `quit` to stop and save the game.

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
# 1. Encode all CSVs, each position tagged with its source (~20s for 16.6M positions) -> data/nofeat.npz (~1.1 GB)
uv run python -m NoFeat.prepare_data
uv run python -m NoFeat.prepare_data --limit 50000 --out data/nofeat_small.npz   # small test set

# 2. Train a named model. --out is required; checkpoints go to <out>/{last,best}.pt
uv run python -m NoFeat.train --out Models/Weights/SlackFishV-T3 --epochs 10
uv run python -m NoFeat.train --out Models/Weights/SlackFishV-T3 --resume Models/Weights/SlackFishV-T3/last.pt --epochs 15
uv run python -m NoFeat.train --out /tmp/smoke --data data/nofeat_small.npz --epochs 2   # smoke test
```

**3. Register it to play.** Add one line to `BOTS` in `Engine/bots.py`:

```python
"t3": NoFeatSpec("Models/Weights/SlackFishV-T3/best.pt"),
```

Then it's available as `--black t3:2` in `play.py` and in the GUI's bot list.

**Choosing data:** `--sources` picks what to train on: `games` (chessData), `random` (random_evals) and `tactics` (tactic_evals). The default is `games random`, matching the original report. Validation is drawn from all three either way and reported per source; sources you didn't train on are marked `*`. Compare models using these per-source numbers: the overall val MSE changes meaning when the mix changes.

```text
epoch 2: train 0.8233  val MSE 0.7999  val MAE 315 cp  mirrored 309 cp  (27s)
  MAE by source: games 273  random 358  tactics* 831   (* = not trained on)
```

**Colour-swap augmentation:** half of the training positions (`--augment-prob 0.5`) are mirrored with colours, castling and turn swapped and the score negated (`NoFeat/augment.py`, equivalent to python-chess `board.mirror()`). `mirrored` in the log is validation MAE on the colour-swapped validation set. If it's far from the normal MAE, the model treats the two colours differently. SlackFishV-T1-3.25M, trained without augmentation, scores 136 cp as-is and 182 cp mirrored.

Useful training options: `--limit N`, `--batch-size` (default 1024), `--lr` (3e-4, warmup then cosine decay), `--d-model`, `--layers`, `--heads`, `--d-ff`, `--clip`, `--scale`, `--wandb`, `--no-compile`. The checkpoint stores the model config and target scale, so `play.py` loads any size without code changes.

**Performance on an RTX 500 Ada laptop GPU (4 GB):** about 3,700 positions/s with `torch.compile`, so one epoch over all 14M positions takes about 62 minutes. Peak GPU memory is about 3 GB at batch size 1024. In play it's much faster than v1/v2, because there's no feature engineering: depth 2 takes about 0.2s per move.

**Smoke test results** (100k positions, 3 epochs): validation MAE fell from 331 to 286 cp. This only shows the pipeline works; no full training run has been done yet.

## Repository Layout

```text
.
├── Engine/
│   ├── bots.py              # Model registry, feature encoding, evaluator, minimax search
│   ├── play.py              # Game runner + CLI (bot/human vs bot/human)
│   ├── gui.py, gui.html     # Browser GUI for human vs bot (Chessground board)
│   ├── export.py            # Game -> animated GIF / final-position PNG
│   ├── GameLoop.py          # Legacy, broken (see below)
│   ├── BotVsBot.py          # Legacy, needs V5 weights
│   ├── startWithBoard.py    # Legacy, broken
│   └── testAllMovesFromFen  # Legacy, broken
├── NoFeat/
│   ├── encode.py            # FEN -> 64 square tokens + castling token + turn token
│   ├── model.py             # ChessTransformerV1
│   ├── prepare_data.py      # CSVs -> data/nofeat.npz
│   ├── train.py             # Training loop
│   └── evaluator.py         # Load a checkpoint for play
├── Models/
│   ├── Scalers/             # scaler_X.pkl, scaler_Y.pkl (v1/v2), scaler_Y_9M.pkl (v5)
│   ├── Training/            # Model definitions and training scripts
│   └── Weights/             # Model checkpoints (V1, V2 only)
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
