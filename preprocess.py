import pandas as pd
import numpy as np
import chess
from tqdm import tqdm

files = [
    "/Users/ankushsarkar/Code/uni/Year4/4al3/project/data/chessData.csv",
    "/Users/ankushsarkar/Code/uni/Year4/4al3/project/data/random_evals.csv",
    "/Users/ankushsarkar/Code/uni/Year4/4al3/project/data/tactic_evals.csv",
]

dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f))
    except pd.errors.EmptyDataError:
        continue

data = pd.concat(dfs, ignore_index=True, sort=False)
print(f"Total number of rows: {len(data)}")

# print(data.head())

# drop move column if it exists
if 'move' in data.columns:
    data = data.drop(columns=['move'])

def fen_to_bitboards(fenstr:str):
    board = chess.Board(fenstr)
    # Mapping piece symbols to chess piece type constants
    piece_type_map = {
        'p': chess.PAWN,
        'n': chess.KNIGHT,
        'b': chess.BISHOP,
        'r': chess.ROOK,
        'q': chess.QUEEN,
        'k': chess.KING
    }
    
    piece_order = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    bitboards = {}

    for piece_symbol in piece_order:
        piece_type = piece_type_map[piece_symbol.lower()]
        color = chess.WHITE if piece_symbol.isupper() else chess.BLACK
        bitboard = board.pieces(piece_type, color)
        bitboards[piece_symbol] = np.array([
            1 if bitboard & chess.BB_SQUARES[i] else 0 for i in range(64)
        ], dtype=np.uint8)

    return bitboards


def piece_count(fen_str: str) -> int:
    score_map = {
        'p': -1,
        'n': -3,
        'b': -3,
        'r': -5,
        'q': -9,
        'P': 1,
        'N': 3,
        'B': 3,
        'R': 5,
        'Q': 9,
    }
    total_score = 0
    for peice in score_map.keys():
        total_score += fen_str.count(peice) * score_map[peice]
    return total_score

def piece_mobility(fen_str: str) -> int:
    board = chess.Board(fen_str)
    cur_color = board.turn
    white_mobility = 0
    black_mobility = 0
    for move in board.legal_moves:
        print(board.san(move))
    board.turn = not board.turn
    print()
    for move in board.legal_moves:
        print(board.san(move))
    return (white_mobility, black_mobility)
    # for 
    # total_mobility = 0
    # for peice in ['b', 'n', 'r', 'q']:
    #     total_mobility += board.pieces(peice, chess.WHITE).count()
    # return total_mobility


preprocessed_data = []
piece_order = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

# for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing positions"):
#     bitboards = fen_to_bitboards(row['FEN'])
#     board_array = np.stack([bitboards[piece] for piece in piece_order], axis=0)
#     eval_str = row['Evaluation']
#     if isinstance(eval_str, str) and '#' in eval_str:
#         # convert mate score to centipawn score
#         mate_in = int(eval_str.split('#')[1])
#         eval_numeric = (32000 - abs(mate_in)) * (1 if mate_in > 0 else -1)
#     else:
#         eval_numeric = float(eval_str)
#     preprocessed_data.append({
#         'bitboards': board_array,
#         'eval': eval_numeric,
#         'piece_count': piece_count(row['FEN']),
#     })

# Convert to structured arrays for easy batching
# X = np.array([d['bitboards'] for d in preprocessed_data])  # shape (N, 12, 64)
# y = np.array([d['eval'] for d in preprocessed_data])  # shape (N,)
# X = np.concatenate([X, piece_count.reshape(-1, 1)], axis=1)  # shape (N, 13)
# # Save to disk
# np.savez_compressed('preprocessed_chess_data.npz', X=X, y=y)
# print(f"\nSaved {len(X)} positions to preprocessed_chess_data.npz")
# print(f"X shape: {X.shape}, y shape: {y.shape}")

piece_mobility(data['FEN'][0])