import pandas as pd
import numpy as np
from tqdm import tqdm
import featureEng

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

# drop move column if it exists
if 'move' in data.columns:
    data = data.drop(columns=['move'])

preprocessed_data = []
piece_order = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing positions"):
    bitboards = featureEng.fen_to_bitboards(row['FEN'])
    board_array = np.stack([bitboards[piece] for piece in piece_order], axis=0)
    eval_str = row['Evaluation']
    if isinstance(eval_str, str) and '#' in eval_str:
        # convert mate score to centipawn score
        mate_in = int(eval_str.split('#')[1])
        eval_numeric = (32000 - abs(mate_in)) * (1 if mate_in > 0 else -1)
    else:
        eval_numeric = float(eval_str)
    preprocessed_data.append({
        'bitboards': board_array, 
        'eval': eval_numeric,
        'piece_count': featureEng.piece_mobility(row['FEN']),
        'piece_mobility': featureEng.piece_mobility(row['FEN']),
        'player_turn': featureEng.player_turn(row['FEN']),
        'en_passant_available': featureEng.en_passant_available(row['FEN']),
        'in_check': featureEng.in_check(row['FEN']),
        'castling_rights': featureEng.castling_rights(row['FEN']),
    })
    # For testing
    # if index == 20:
    #     break


# Save to disk
np.savez_compressed(
    "preprocessed_chess_data.npz",
    bitboards=np.array([d['bitboards'] for d in preprocessed_data]),
    eval=np.array([d['eval'] for d in preprocessed_data]),
    piece_count=np.array([d['piece_count'] for d in preprocessed_data]),
    piece_mobility=np.array([d['piece_mobility'] for d in preprocessed_data]),
    player_turn=np.array([d['player_turn'] for d in preprocessed_data]),
    en_passant_available=np.array([d['en_passant_available'] for d in preprocessed_data]),
    in_check=np.array([d['in_check'] for d in preprocessed_data]),
    castling_rights=np.array([d['castling_rights'] for d in preprocessed_data]),
)