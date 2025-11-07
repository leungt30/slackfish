import chess
import numpy as np
import pandas as pd
import random

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

def predict(bitboard: np.array):
    # Dummy model prediction
    return random.uniform(-10, 10)

# Placeholder scoring function
# would take in a board in fen format
# and return a score from the model
def get_score(board: chess.Board, move: chess.Move):
    temp_board = board.copy()
    temp_board.push(move)
    bitboard = fen_to_bitboards(temp_board.fen())
    return predict(bitboard)





board = chess.Board()
print(board)
print("------------------------")
print()


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

while not board.is_game_over():
    if board.turn == player_color:
        # player's turn
        user_move = input("Enter your move in UCI format (e.g., e2e4): ")

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
            score = get_score(board, move)
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
            
        
        