import featureEng
import chess
import chess
import joblib
import numpy as np
import pandas as pd
import random
import torch

import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.append("Models/Training")

from Models import SlackFishCNN_V5


piece_order = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]


def fen_to_bitboards(fenstr: str):
    board = chess.Board(fenstr)
    # Mapping piece symbols to chess piece type constants
    piece_type_map = {
        "p": chess.PAWN,
        "n": chess.KNIGHT,
        "b": chess.BISHOP,
        "r": chess.ROOK,
        "q": chess.QUEEN,
        "k": chess.KING,
    }

    bitboards = {}

    for piece_symbol in piece_order:
        piece_type = piece_type_map[piece_symbol.lower()]
        color = chess.WHITE if piece_symbol.isupper() else chess.BLACK
        bitboard = board.pieces(piece_type, color)
        bitboards[piece_symbol] = np.array(
            [1 if bitboard & chess.BB_SQUARES[i] else 0 for i in range(64)],
            dtype=np.uint8,
        )

    return bitboards




with np.load("preprocessed_chess_data.npz") as data:
    print(data.files)
    for f in data.files:
        print(f"{f} : {data[f].shape}")

    arrays = [
        data["piece_count"][:,None],
        data["player_turn"][:, None],
        data["en_passant_available"][:, None],
        data["in_check"][:, None],
        data["castling_rights"],
        data["pst_score"][:, None],
        data["check_one_move_away"],
        data["legal_moves_per_side"],
        data["is_forking"], # disable for partial
        data['bishop_activity'],
        data['pinned_pieces'],
        data['value_of_hanging_pieces'],
        data['center_control'],
        data['is_win'][:,None]
    ]
    Xs = np.concatenate(arrays, axis=1)
    y = data['eval']

    indices = np.arange(len(Xs))

    y_train = y.reshape(-1,1)
    raw = data['bitboards'][0]
    raw_train = torch.tensor(raw, dtype=torch.float32).reshape(-1, 12, 8, 8)

    iso_train = torch.tensor(data['isolated_pawns'][0],
                             dtype=torch.float32).reshape(-1, 2, 8, 8)
    dbl_train = torch.tensor(data['double_pawns'][0],
                             dtype=torch.float32).reshape(-1, 2, 8, 8)
    rook7_train = torch.tensor(data['rook_on_7th_rank'][0],
                               dtype=torch.float32).reshape(-1, 2, 8, 8)
    
    semi_open_train = torch.tensor(data['rook_on_semi_open_file'][0],
                                   dtype=torch.float32).reshape(-1, 2, 8, 8)
    
    same_file_train = torch.tensor(data['rooks_on_same_file'][0],
                                   dtype=torch.float32).reshape(-1, 2, 8, 8)
    
    hanging_train = torch.tensor(data['hanging_pieces_bitboards'][0],
                                 dtype=torch.float32).reshape(-1, 2, 8, 8)

    board_train = torch.cat([
        raw_train,
        iso_train,
        dbl_train,
        rook7_train,
        semi_open_train,
        same_file_train,
        hanging_train
    ], dim=1)

    x_train_t = torch.tensor(Xs, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    print(board_train.shape, x_train_t.shape, y_train_t.shape)


# Create objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = x_train_t.shape[1]
num_bitboards = board_train[0].shape[0]
# should be 24, 24
model = SlackFishCNN_V5(input_feat = num_features, num_bitboards = num_bitboards).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    eps=1e-8,
    weight_decay=1e-4,
)

# Load scalers
Scaler_X = joblib.load("Models/Scalers/scaler_X.pkl")
Scaler_Y = joblib.load("Models/Scalers/scaler_Y_9M.pkl")

# Load model weights
model_path = "Models/Weights/SlackFishCNN_V5_79"
checkpoint = torch.load(model_path, map_location="cuda")

# Load model state
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model.eval()
print("Model loaded.")
print("=================================")
print()


def normalize(data_x):
    x_scaled = Scaler_X.fit_transform(data_x)

    return x_scaled


def predict(fenStr: str):
    bitboards = fen_to_bitboards(fenStr)
    board_array = np.stack([bitboards[piece] for piece in piece_order], axis=0)
     
    features = np.array(
        [
            featureEng.piece_count(fenStr),
            featureEng.player_turn(fenStr),
            featureEng.en_passant_available(fenStr),
            featureEng.in_check(fenStr),
            *featureEng.castling_rights(fenStr),
            featureEng.pst_score(fenStr),
            *featureEng.check_one_move_away(fenStr),
            *featureEng.legal_moves_per_side(fenStr),
            *featureEng.is_forking(fenStr),
            *featureEng.bishop_activity(fenStr),
            *featureEng.pinned_pieces(fenStr),
            *featureEng.value_of_hanging_pieces(fenStr),
            *featureEng.center_control(fenStr),
            featureEng.is_win(fenStr),
        ]
    ).reshape(1, -1)

    boards_array = torch.cat([
        torch.tensor(board_array, dtype=torch.float32).reshape(-1, 12, 8, 8),
        torch.tensor(featureEng.isolated_pawns(fenStr),
                     dtype=torch.float32).reshape(-1, 2, 8, 8),
        torch.tensor(featureEng.double_pawns(fenStr),
                     dtype=torch.float32).reshape(-1, 2, 8, 8),
        torch.tensor(featureEng.rook_on_7th_rank(fenStr),
                     dtype=torch.float32).reshape(-1, 2, 8, 8),
        torch.tensor(featureEng.rook_on_semi_open_file(fenStr),
                     dtype=torch.float32).reshape(-1, 2, 8, 8),
        torch.tensor(featureEng.rooks_on_same_file(fenStr),
                     dtype=torch.float32).reshape(-1, 2, 8, 8),
        torch.tensor(featureEng.hanging_pieces_bitboards(fenStr),
                     dtype=torch.float32).reshape(-1, 2, 8, 8),
    ], dim=1)

    # New model doesn't normalize x
    # features = normalize(features)

    features_t = torch.tensor(features, dtype=torch.float32)

    score_t = model(boards_array.to(device), features_t.to(device))
    score = Scaler_Y.inverse_transform(score_t.cpu().detach().numpy())
    return score

# fens = [
#     "r1bqr1k1/ppp2ppp/2n2n2/3p4/1b1P4/2NB1N2/PPP1QPPP/R1B2RK1 b - - 1 10", # free queen for rook 
#     "r1b1r1k1/ppp2ppp/2n5/3pN3/1b1P2nq/2NB4/PPPBQPPP/R4RK1 b - - 5 12", # mate in 1 Qxh2
#     "r1b1rqk1/ppp2ppp/2n2n2/1N1p4/1b1P4/3B1N2/PPPB1PPP/R2R1QK1 w - - 20 20" # Nxc7 fork
#     "r1b2rk1/ppb2ppp/2n1p3/3n4/2NP3q/2NB4/PP1B1PPP/R2Q1RK1 w - - 10 13" # white needs to prevent mate in 1
#     "r1b2rk1/ppb2ppp/2n1pq2/3n4/2NP1R2/2NB4/PP3PPP/R2QB1K1 b - - 19 17" # capture the rook one of 3 ways
#     "6k1/3r2p1/3pR1Bp/1bp5/8/7P/5KP1/8 w - - 0 1" # Re8 mate
#     "7r/2K3PP/1P1B1P2/2Pp4/2R5/3P1Np1/5p1p/r4k2 b - - 0 1" # Ra2 mate
#     "8/1P4pp/8/8/2K5/8/P5pk/8 b HAha - 0 1" # promote pawn
#     "r4r1k/2p4p/n5p1/2b1p1n1/2B1b1P1/7P/PBP2P2/4RRK1 w q - 0 1" # Bxe5 mate
#     "r4rk1/2p1qppp/n7/2b1p1n1/6P1/1B4KP/PBPQ1P2/1N1R2R1 b - - 10 6" # Ke4 fork
# ]
fen_str = "r1b1r1k1/ppp2ppp/2n5/3pN3/1b1P2nq/2NB4/PPPBQPPP/R4RK1 b - - 5 12"
board = chess.Board(fen_str)
print(board)

board.turn = chess.WHITE if fen_str.split()[1] == 'w' else chess.BLACK
best_move = None
best_score = -9999999
for move in board.legal_moves:
    temp_board = board.copy()
    temp_board.push(move)
    score = predict(temp_board.fen()) * 1 if board.turn == chess.WHITE else -1
    if score > best_score:
        best_score = score
        best_move = move
    print(f"{temp_board.piece_at(move.to_square)} {move} {score}")

print(f"Best Move: {best_move} with score {best_score}")
print("=================================")
print(board.fen())
print(board)

# print("------------------- Feature Extraction ------------------")
# print("Piece Count:")
# print(featureEng.piece_count(fen_str))
# print("--------------------------------------------------")
# print("Player Turn:")
# print(featureEng.player_turn(fen_str))
# print("--------------------------------------------------")
# print("En Passant Available:")
# print(featureEng.en_passant_available(fen_str))
# print("--------------------------------------------------")
# print("In Check:")
# print(featureEng.in_check(fen_str))
# print("--------------------------------------------------")
# print("Castling Rights:")
# print(featureEng.castling_rights(fen_str))
# print("--------------------------------------------------")
# print("PST Score:")
# print(featureEng.pst_score(fen_str))
# print("--------------------------------------------------")
# print("Check One Move Away:")
# print(featureEng.check_one_move_away(fen_str))
# print("--------------------------------------------------")
# print("Legal Moves Per Side:")
# print(featureEng.legal_moves_per_side(fen_str))
# print("--------------------------------------------------")
# print("Isolated Pawns:")
# print(featureEng.isolated_pawns(fen_str))
# print("--------------------------------------------------")
# print("Double Pawns:")
# print(featureEng.double_pawns(fen_str))
# print("--------------------------------------------------")
# print("Rook on 7th Rank:")
# print(featureEng.rook_on_7th_rank(fen_str))
# print("--------------------------------------------------")
# print("Rook on Semi-Open File:")
# print(featureEng.rook_on_semi_open_file(fen_str))
# print("--------------------------------------------------")
# print("Rooks on Same File:")
# print(featureEng.rooks_on_same_file(fen_str))
# print("--------------------------------------------------")
# print("Is Forking:")
# print(featureEng.is_forking(fen_str))
# print("--------------------------------------------------")
# print("Fork Available:")
# print(featureEng.fork_available(fen_str))
# print("--------------------------------------------------")
# print("Bishop Activity:")
# print(featureEng.bishop_activity(fen_str))
# print("--------------------------------------------------")
# print("Pinned Pieces:")
# print(featureEng.pinned_pieces(fen_str))
# print("--------------------------------------------------")
# print("Value of Hanging Pieces:")
# print(featureEng.value_of_hanging_pieces(fen_str))
# print("--------------------------------------------------")
# print("Hanging Pieces Bitboards:")
# print(featureEng.hanging_pieces_bitboards(fen_str))
# print("--------------------------------------------------")
# print("Center Control:")
# print(featureEng.center_control(fen_str))
# print("--------------------------------------------------")
# print("Is Win:")
# print(featureEng.is_win(fen_str))
