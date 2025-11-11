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

from SlackFishCNN import SlackFishCNN_V1
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

def predict(fenStr: str):
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
def get_score(board: chess.Board, move: chess.Move):
    moved_board = board.copy()
    moved_board.push(move)
    fenStr = moved_board.fen()
    return predict(fenStr)

# Create objects
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SlackFishCNN_V1(11).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    eps=1e-8,
    weight_decay=1e-4,
)

# Load scalers
Scaler_X = joblib.load("Models/Scalers/scaler_X.pkl")
Scaler_Y = joblib.load("Models/Scalers/scaler_Y.pkl")

# Load model weights
model_path = "Models/Weights/SlackFishCNN_V1_499"
checkpoint = torch.load(model_path, map_location="cuda")

# Load model state
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model.eval()
print("Model loaded.")

# Debug prints
# print(chess.Board("r1b1kb1r/1p1pn2p/p3p3/1p3pP1/3P1P1p/Pn2P3/NPP1K3/1RBQ4 w kq - 2 19"))
# print(predict("r1b1kb1r/1p1pn2p/p3p3/1p3pP1/3P1P1p/Pn2P3/NPP1K3/1RBQ4 w kq - 2 19"))
# print(featureEng.pst_score("r1b1kb1r/1p1pn2p/p3p3/1p3pP1/3P1P1p/Pn2P3/NPP1K3/1RBQ4 w kq - 2 19"))
# print(featureEng.piece_count("r1b1kb1r/1p1pn2p/p3p3/1p3pP1/3P1P1p/Pn2P3/NPP1K3/1RBQ4 w kq - 2 19"))
# print(featureEng.piece_mobility("r1b1kb1r/1p1pn2p/p3p3/1p3pP1/3P1P1p/Pn2P3/NPP1K3/1RBQ4 w kq - 2 19"))
# print("========================")


# Create board
board = chess.Board()
print(board)
print("------------------------")
print()

# Select color and start game loop
print("Select white or black pieces to play as:")
color = None
while color == None:
    color = input("Enter 'w' for white or 'b' for black: ").strip().lower()
    if color not in ['w', 'b']:
        print("Invalid input, try again.")
        color = None

if color == 'w':
    player_color = chess.WHITE
else:
    player_color = chess.BLACK

# Game loop
while not board.is_game_over():
    if board.turn == player_color:
        # player's turn
        user_move = input("Enter your move in UCI format (e.g., e2e4): ")

        
        if " " in user_move or len(user_move) != 4 and len(user_move) != 5:
            print("Invalid move format. Try again.")
            print()
        else:
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                board.push(move)
                print("========================")
                print("you played:", move)
                print("------------------------")
                print(board)
                print("========================")
                print()
            else:
                print("Illegal move. Try again.")
                print()
    else:
        # computer turn
        # uses random move for now
        board.generate_legal_moves()
        best_move = None
        best_score = -float('inf')
        print("Computer is thinking...")
        for move in board.legal_moves:
            score = get_score(board, move) * (-1 if color == "w" else 1)
            # Print to show all moves and score
            print(f"Move: {move}, Score: {score}")
            if score > best_score:
                best_score = score
                best_move = move
        
        board.push(best_move)
        print("========================")
        print("computer played:", best_move)
        print("------------------------")
        print(board)
        print("========================")
        print()
            
        
        