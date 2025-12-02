import pandas as pd
import numpy as np
from tqdm import tqdm
import featureEng

files = [
    "data/chessData.csv",
    "data/random_evals.csv",
    "data/tactic_evals.csv",
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
        'piece_count': featureEng.piece_count(row['FEN']),
        'piece_mobility': featureEng.piece_mobility(row['FEN']),
        'player_turn': featureEng.player_turn(row['FEN']),
        'en_passant_available': featureEng.en_passant_available(row['FEN']),
        'in_check': featureEng.in_check(row['FEN']),
        'castling_rights': featureEng.castling_rights(row['FEN']),
        'pst_score': featureEng.pst_score(row['FEN']),
        'check_one_move_away': featureEng.check_one_move_away(row['FEN']),
        'legal_moves_per_side': featureEng.legal_moves_per_side(row['FEN']),
        'issolated_pawns': featureEng.issolated_pawns(row['FEN']),
        'double_pawns': featureEng.double_pawns(row['FEN']),
        'rook_on_7th_rank': featureEng.rook_on_7th_rank(row['FEN']),
        'rook_on_semi_open_file': featureEng.rook_on_semi_open_file(row['FEN']),
        'rooks_on_same_file': featureEng.rooks_on_same_file(row['FEN']),
        'is_forking': featureEng.is_forking(row['FEN']),
        'fork_available': featureEng.fork_available(row['FEN']),
        'bishop_activity': featureEng.bishop_activity(row['FEN']),
        'pinned_pieces': featureEng.pinned_pieces(row['FEN']),
        'number_of_hanging_pieces': featureEng.number_of_hanging_pieces(row['FEN']),
        'hanging_pieces_bitboards': featureEng.hanging_pieces_bitboards(row['FEN']),
        'center_controll': featureEng.center_controll(row['FEN']),
    })

    # Uncomment for testing with a subset of data
    if index == 20:
        break


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
    pst_score=np.array([d['pst_score'] for d in preprocessed_data]),
    check_one_move_away=np.array([d['check_one_move_away'] for d in preprocessed_data]),
    legal_moves_per_side=np.array([d['legal_moves_per_side'] for d in preprocessed_data]),
    issolated_pawns=np.array([d['issolated_pawns'] for d in preprocessed_data]),
    double_pawns=np.array([d['double_pawns'] for d in preprocessed_data]),
    rook_on_7th_rank=np.array([d['rook_on_7th_rank'] for d in preprocessed_data]),
    rook_on_semi_open_file=np.array([d['rook_on_semi_open_file'] for d in preprocessed_data]),
    rooks_on_same_file=np.array([d['rooks_on_same_file'] for d in preprocessed_data]),
    is_forking=np.array([d['is_forking'] for d in preprocessed_data]),
    fork_available=np.array([d['fork_available'] for d in preprocessed_data]),
    bishop_activity=np.array([d['bishop_activity'] for d in preprocessed_data]),
    pinned_pieces=np.array([d['pinned_pieces'] for d in preprocessed_data]),
    number_of_hanging_pieces=np.array([d['number_of_hanging_pieces'] for d in preprocessed_data]),
    hanging_pieces_bitboards=np.array([d['hanging_pieces_bitboards'] for d in preprocessed_data]),
    center_controll=np.array([d['center_controll'] for d in preprocessed_data]),
)

