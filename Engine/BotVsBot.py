import chess
import joblib
import numpy as np
import pandas as pd
import random
import chess.pgn
import torch
import torch.nn as nn
import time
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.append("Models/Training")

from SlackFishCNN import SlackFishCNN_V1, SlackFishCNN_V2
import featureEng


piece_order = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

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
    
    bitboards = {}

    for piece_symbol in piece_order:
        piece_type = piece_type_map[piece_symbol.lower()]
        color = chess.WHITE if piece_symbol.isupper() else chess.BLACK
        bitboard = board.pieces(piece_type, color)
        bitboards[piece_symbol] = np.array([
            1 if bitboard & chess.BB_SQUARES[i] else 0 for i in range(64)
        ], dtype=np.uint8)

    return bitboards

def normalize(data_x):
    x_scaled = Scaler_X.fit_transform(data_x)
    
    return x_scaled

def predict(model:nn.Module, fenStr: str):
    bitboards = fen_to_bitboards(fenStr)
    board_array = np.stack([bitboards[piece] for piece in piece_order], axis=0)
    features = np.array([
        featureEng.piece_count(fenStr),
        *featureEng.piece_mobility(fenStr),
        featureEng.player_turn(fenStr),
        featureEng.en_passant_available(fenStr),
        featureEng.in_check(fenStr),
        *featureEng.castling_rights(fenStr),
        featureEng.pst_score(fenStr),
    ]).reshape(1, -1)
    features = normalize(features)

    bitboards_t = torch.tensor(board_array, dtype=torch.float32).reshape(-1, 12, 8, 8)
    features_t = torch.tensor(features, dtype=torch.float32)

    score_t = model(bitboards_t.to(device), features_t.to(device))
    score = Scaler_Y.inverse_transform(score_t.cpu().detach().numpy())
    return score

# Scoring function
def get_score(model:nn.Module, board: chess.Board, move: chess.Move):
    moved_board = board.copy()
    moved_board.push(move)
    fenStr = moved_board.fen()
    return predict(model, fenStr)

# Create objects
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_white = SlackFishCNN_V1(11).to(device)
# optimizer = torch.optim.Adam(
#     model_white.parameters(),
#     lr=0.001,
#     eps=1e-8,
#     weight_decay=1e-4,
# )
model_black = SlackFishCNN_V2(11).to(device)

# Load scalers
Scaler_X = joblib.load("Models/Scalers/scaler_X.pkl")
Scaler_Y = joblib.load("Models/Scalers/scaler_Y.pkl")

# Load model weights
model_path_white = "Models/Weights/SlackFishCNN_V1_499"
model_path_black = "Models/Weights/SlackFishCNN_V2_449"

checkpoint_white = torch.load(model_path_white, map_location="cuda")
checkpoint_black = torch.load(model_path_black, map_location="cuda")

# Load model state
model_white.load_state_dict(checkpoint_white["model_state_dict"])
model_black.load_state_dict(checkpoint_black["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model_white.eval()
model_black.eval()
print("Model loaded.")


# Create board
board = chess.Board()
game = chess.pgn.Game()
node = game
print(board)
print("------------------------")
print()
models = [model_white, model_black]
# Select color and start game loop
# Game loop
while not board.is_game_over():
    # computer turn
    board.generate_legal_moves()
    best_move = None
    best_score = -float('inf') 
    # print("Computer 1 is thinking...")
    for move in board.legal_moves:
        score = get_score(models[0 if board.turn else 1], board, move) * (1 if board.turn == chess.WHITE else -1)
        # Print to show all moves and score
        # print(f"Move: {move}, Score: {score}")
        if score > best_score:
            best_score = score
            best_move = move
        
    board.push(best_move)
    node = node.add_variation(best_move)
    # print("========================")
    # print(f"Computer {1 if board.turn else 2} played:", best_move)
    # print("------------------------")
    # print(board)
    # print("========================")
    # print()



            
        
with open("game.pgn", "w") as f:
    print(game, file=f)
print("EOF")