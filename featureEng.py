import numpy as np
import chess

def center_control(bitboards):
    return np.sum(bitboards)

def pawn_structure(bitboards):
    return np.sum(bitboards)

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
    white_mobility = 0
    black_mobility = 0
    for move in board.legal_moves:
        if board.turn == chess.WHITE:
            white_mobility += 1
        else:
            black_mobility += 1
    board.turn = not board.turn
    for move in board.legal_moves:
        if board.turn == chess.WHITE:
            white_mobility += 1
        else:
            black_mobility += 1
    return (white_mobility, black_mobility)
