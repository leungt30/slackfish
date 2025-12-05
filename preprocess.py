import pandas as pd
import numpy as np
from tqdm import tqdm
import featureEng
import os
import signal

files = [
    "data/chessData.csv",
    "data/random_evals.csv",
    # "data/tactic_evals.csv",
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
if "move" in data.columns:
    data = data.drop(columns=["move"])

output_file = "preprocessed_chess_data.npz"
# Save checkpoint every N rows (adjust based on your needs)
SAVE_INTERVAL = 1000  # Save every 1000 rows

# Delete existing file to start fresh
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Deleted existing {output_file}")

# Initialize empty lists
all_bitboards = []
all_eval = []
all_piece_count = []
all_piece_mobility = []
all_player_turn = []
all_en_passant_available = []
all_in_check = []
all_castling_rights = []
all_pst_score = []
all_check_one_move_away = []
all_legal_moves_per_side = []
all_isolated_pawns = []
all_double_pawns = []
all_rook_on_7th_rank = []
all_rook_on_semi_open_file = []
all_rooks_on_same_file = []
all_is_forking = []
all_fork_available = []
all_bishop_activity = []
all_pinned_pieces = []
all_value_of_hanging_pieces = []
all_hanging_pieces_bitboards = []
all_center_control = []

piece_order = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]


def save_data():
    """Save all accumulated data to disk"""
    np.savez_compressed(
        output_file,
        bitboards=np.array(all_bitboards),
        eval=np.array(all_eval),
        piece_count=np.array(all_piece_count),
        piece_mobility=np.array(all_piece_mobility),
        player_turn=np.array(all_player_turn),
        en_passant_available=np.array(all_en_passant_available),
        in_check=np.array(all_in_check),
        castling_rights=np.array(all_castling_rights),
        pst_score=np.array(all_pst_score),
        check_one_move_away=np.array(all_check_one_move_away),
        legal_moves_per_side=np.array(all_legal_moves_per_side),
        isolated_pawns=np.array(all_isolated_pawns),
        double_pawns=np.array(all_double_pawns),
        rook_on_7th_rank=np.array(all_rook_on_7th_rank),
        rook_on_semi_open_file=np.array(all_rook_on_semi_open_file),
        rooks_on_same_file=np.array(all_rooks_on_same_file),
        is_forking=np.array(all_is_forking),
        fork_available=np.array(all_fork_available),
        bishop_activity=np.array(all_bishop_activity),
        pinned_pieces=np.array(all_pinned_pieces),
        value_of_hanging_pieces=np.array(all_value_of_hanging_pieces),
        hanging_pieces_bitboards=np.array(all_hanging_pieces_bitboards),
        center_control=np.array(all_center_control),
    )

def signal_handler(signum, frame):
    """Handle signals (SIGTERM, SIGINT) by saving data before exit"""
    print(f"\n\nReceived signal {signum}. Saving progress before exit...")
    save_data()
    print(f"Progress saved! Processed {len(all_eval)} rows before exit.")
    exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)  # For kill command
signal.signal(signal.SIGINT, signal_handler)   # For Ctrl+C

for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing positions"):
    try:
        bitboards = featureEng.fen_to_bitboards(row["FEN"])
        board_array = np.stack([bitboards[piece] for piece in piece_order], axis=0)
        eval_str = row["Evaluation"]
        if isinstance(eval_str, str) and "#" in eval_str:
            # convert mate score to centipawn score
            mate_in = int(eval_str.split("#")[1])
            eval_numeric = (32000 - abs(mate_in)) * (1 if mate_in > 0 else -1)
        else:
            eval_numeric = float(eval_str)

        all_bitboards.append(board_array)
        all_eval.append(eval_numeric)
        all_piece_count.append(featureEng.piece_count(row["FEN"]))
        all_piece_mobility.append(featureEng.piece_mobility(row["FEN"]))
        all_player_turn.append(featureEng.player_turn(row["FEN"]))
        all_en_passant_available.append(featureEng.en_passant_available(row["FEN"]))
        all_in_check.append(featureEng.in_check(row["FEN"]))
        all_castling_rights.append(featureEng.castling_rights(row["FEN"]))
        all_pst_score.append(featureEng.pst_score(row["FEN"]))
        all_check_one_move_away.append(featureEng.check_one_move_away(row["FEN"]))
        all_legal_moves_per_side.append(featureEng.legal_moves_per_side(row["FEN"]))
        all_isolated_pawns.append(featureEng.isolated_pawns(row["FEN"]))
        all_double_pawns.append(featureEng.double_pawns(row["FEN"]))
        all_rook_on_7th_rank.append(featureEng.rook_on_7th_rank(row["FEN"]))
        all_rook_on_semi_open_file.append(featureEng.rook_on_semi_open_file(row["FEN"]))
        all_rooks_on_same_file.append(featureEng.rooks_on_same_file(row["FEN"]))
        all_is_forking.append(featureEng.is_forking(row["FEN"]))
        all_fork_available.append(featureEng.fork_available(row["FEN"]))
        all_bishop_activity.append(featureEng.bishop_activity(row["FEN"]))
        all_pinned_pieces.append(featureEng.pinned_pieces(row["FEN"]))
        all_value_of_hanging_pieces.append(
            featureEng.value_of_hanging_pieces(row["FEN"])
        )
        all_hanging_pieces_bitboards.append(
            featureEng.hanging_pieces_bitboards(row["FEN"])
        )
        all_center_control.append(featureEng.center_control(row["FEN"]))

        # Save in batches to avoid frequent I/O
        if (index + 1) % SAVE_INTERVAL == 0:
            save_data()
            print(f"\nCheckpoint saved at row {index + 1}")

    except Exception as e:
        print(f"\nError processing row {index}: {e}")
        # Save progress before continuing in case of crash
        save_data()
        print(f"Progress saved. Skipping row {index} and continuing...")
        continue

    # Uncomment for testing with a subset of data
    # if index == 20:
    #     break

# Final save to ensure everything is saved
save_data()
print(f"\nProcessing complete! Saved {len(all_eval)} rows to {output_file}")
