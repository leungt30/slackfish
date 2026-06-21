# SlackFish

SlackFish is a lightweight chess evaluation engine that uses a neural network to approximate Stockfish position evaluations. It converts chess positions from FEN into bitboards and engineered chess features, predicts a centipawn-style evaluation, and uses that score to choose moves.

The project was built under limited compute constraints, so the focus is on combining learned board representations with explicit chess knowledge such as material balance, mobility, castling rights, piece-square tables, hanging pieces, pawn structure, pins, forks, and center control.

## Project Overview

SlackFish trains CNN-based regressors on labeled chess positions. The labels come from Stockfish evaluations rather than human move choices, so the model learns to evaluate board states rather than directly predict a move.

The latest reported model uses:

- 12 piece bitboards plus additional feature bitboards for pawn structure, rook placement, and hanging pieces
- scalar engineered features for material, turn, en passant, check, castling rights, PST score, legal moves, forks, pins, bishop activity, hanging-piece value, center control, and win state
- convolutional layers over board-state channels
- residual blocks in later model versions
- fully connected layers that combine CNN output with engineered features
- MSE loss with the Adam optimizer

## Repository Layout

```text
.
├── Engine/
│   ├── GameLoop.py          # Human-vs-engine command-line game loop
│   ├── BotVsBot.py          # Model-vs-model self-play script
│   ├── startWithBoard.py    # Start/evaluate from a specific board state
│   └── testAllMovesFromFen  # Score legal moves from a FEN position
├── Models/
│   ├── Scalers/             # Saved sklearn/joblib scalers
│   ├── Training/            # Model definitions and training scripts
│   └── Weights/             # Saved model checkpoints
├── featureEng.py            # FEN-to-feature engineering utilities
├── preprocess.py            # Dataset preprocessing into .npz format
├── test.py                  # Quick inspection for preprocessed data
└── REPORT.md                # Original final report
```

## Data

The report describes three candidate datasets:

- Stockfish 11 evaluations at depth 22
- random-move positions from official databases evaluated by Stockfish 12 at depth 20
- Lichess tactics positions

The final training setup used the first two sources and dropped the tactics subset because tactics often contain forced move sequences rather than general position-evaluation examples. The final processed training subset contained about 9 million positions out of the available 14 million due to hardware and memory limits.

`preprocess.py` expects source CSV files under `data/`:

```text
data/chessData.csv
data/random_evals.csv
```

Those raw data files are not included in this repository.

## Features

SlackFish uses a mix of board channels and scalar features.

Board-style features include:

- piece bitboards for each piece type and color
- isolated pawns
- doubled pawns
- rooks on the 7th rank
- rooks on semi-open files
- rooks on the same file
- hanging pieces

Scalar features include:

- material count
- legal moves per side
- side to move
- en passant availability
- current check status
- castling rights
- piece-square-table score
- one-move check availability
- pinned pieces
- hanging-piece value
- center control
- knight forks and fork availability
- bishop activity
- terminal win state

## Setup

Use Python 3.12 or a compatible Python 3 version with PyTorch installed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas python-chess joblib scikit-learn torch tqdm
```

If you only want to run inference with the included scripts, make sure the expected model artifacts exist:

```text
Models/Scalers/scaler_X.pkl
Models/Scalers/scaler_Y.pkl
Models/Scalers/scaler_Y_9M.pkl
Models/Weights/SlackFishCNN_V1_499
Models/Weights/SlackFishCNN_V2_449
```

Some scripts reference newer V5 weights. If you use those paths, make sure the matching checkpoint is available under `Models/Weights/`.

## Usage

Run a human-vs-engine game from the starting position:

```bash
python Engine/GameLoop.py
```

Run model-vs-model self-play and write a PGN:

```bash
python Engine/BotVsBot.py
```

Preprocess raw CSV data into `preprocessed_chess_data.npz`:

```bash
python preprocess.py
```

Inspect the preprocessed dataset:

```bash
python test.py
```

Moves are entered in UCI format, for example:

```text
e2e4
g1f3
e7e8q
```

## Evaluation

The report used an 80/20 train-test split with random shuffling each epoch. Cross-validation was not used because the dataset was already very large, with roughly 2 million samples in the test split.

The main metric was mean squared error against Stockfish evaluations. Accuracy was not used because SlackFish predicts a board evaluation, not a move class.

For qualitative evaluation, the engine also compared its top-scoring moves against Stockfish's top moves on selected board states. The latest reported model scored 8/10 on a small tactical scenario set, compared with 3.5/10 for an older model.

## Known Limitations

SlackFish is strongest in early and midgame positions where material, activity, and tactical features are informative. The report identifies several persistent weaknesses:

- endgame evaluation is less reliable
- checkmate sequences are difficult because they often require exact multi-move planning
- deterministic move selection can lead to repeated positions and draws
- FEN inputs do not encode game history, so the model cannot directly learn threefold-repetition avoidance

Future improvements suggested in the report include endgame-specific features, a dedicated endgame model, stronger checkmate-oriented features, and adjusted mate-score scaling.

## References

The project was influenced by Stockfish-style evaluation, bitboard representations, piece-square tables, Maia Chess, DeepChess/AlphaZero-style hybrid approaches, and ResNet residual connections. See `REPORT.md` for the full discussion and source links.
