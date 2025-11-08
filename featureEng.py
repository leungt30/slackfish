import numpy as np
import chess

def center_control(bitboards):
    # Placeholder implementation
    return np.sum(bitboards)

def pawn_structure(bitboards):
    # Placeholder implementation
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

def player_turn(fen_str: str) -> int:
    board = chess.Board(fen_str)
    return 1 if board.turn == chess.WHITE else 0

def en_passant_available(fen_str: str) -> int:
    board = chess.Board(fen_str)
    return 1 if board.has_legal_en_passant() else 0

def in_check(fen_str: str) -> int:
    board = chess.Board(fen_str)
    return 1 if board.is_check() else 0

def castling_rights(fen_str: str) -> int:
    board = chess.Board(fen_str)
    white_king_side = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    white_queen_side = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    black_king_side = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    black_queen_side = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    return (white_king_side, white_queen_side, black_king_side, black_queen_side)

def pst_score(fenstr: str) -> int:
    values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
    pst = {
        'P': (   0,   0,   0,   0,   0,   0,   0,   0,
                78,  83,  86,  73, 102,  82,  85,  90,
                 7,  29,  21,  44,  40,  31,  44,   7,
               -17,  16,  -2,  15,  14,   0,  15, -13,
               -26,   3,  10,   9,   6,   1,   0, -23,
               -22,   9,   5, -11, -10,  -2,   3, -19,
               -31,   8,  -7, -37, -36, -14,   3, -31,
                 0,   0,   0,   0,   0,   0,   0,   0),
        'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
                -3,  -6, 100, -36,   4,  62,  -4, -14,
                10,  67,   1,  74,  73,  27,  62,  -2,
                24,  24,  45,  37,  33,  41,  25,  17,
                -1,   5,  31,  21,  22,  35,   2,   0,
               -18,  10,  13,  22,  18,  15,  11, -14,
               -23, -15,   2,   0,   2,   0, -23, -20,
               -74, -23, -26, -24, -19, -35, -22, -69),
        'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
               -11,  20,  35, -42, -39,  31,   2, -22,
                -9,  39, -32,  41,  52, -10,  28, -14,
                25,  17,  20,  34,  26,  25,  15,  10,
                13,  10,  17,  23,  17,  16,   0,   7,
                14,  25,  24,  15,   8,  25,  20,  15,
                19,  20,  11,   6,   7,   6,  20,  16,
                -7,   2, -15, -12, -14, -15, -10, -10),
        'R': (  35,  29,  33,   4,  37,  33,  56,  50,
                55,  29,  56,  67,  55,  62,  34,  60,
                19,  35,  28,  33,  45,  27,  25,  15,
                 0,   5,  16,  13,  18,  -4,  -9,  -6,
               -28, -35, -16, -21, -13, -29, -46, -30,
               -42, -28, -42, -25, -25, -35, -26, -46,
               -53, -38, -31, -26, -29, -43, -44, -53,
               -30, -24, -18,   5,  -2, -18, -31, -32),
        'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
                14,  32,  60, -10,  20,  76,  57,  24,
                -2,  43,  32,  60,  72,  63,  43,   2,
                 1, -16,  22,  17,  25,  20, -13,  -6,
               -14, -15,  -2,  -5,  -1, -10, -20, -22,
               -30,  -6, -13, -11, -16, -11, -16, -27,
               -36, -18,   0, -19, -15, -15, -21, -38,
               -39, -30, -31, -13, -31, -36, -34, -42),
        'K': (   4,  54,  47, -99, -99,  60,  83, -62,
               -32,  10,  55,  56,  56,  55,  10,   3,
               -62,  12, -57,  44, -67,  28,  37, -31,
               -55,  50,  11,  -4, -19,  13,   0, -49,
               -55, -43, -52, -28, -51, -47,  -8, -50,
               -47, -42, -43, -79, -64, -32, -29, -32,
                -4,   3, -14, -50, -57, -18,  13,   4,
                17,  30,  -3, -14,   6,  -1,  40,  18),
    }
    score = 0
    position = 0
    for char in fenstr.split(" ")[0]:
        if char.upper() in pst.keys():
            # uses position to get value from pst table if white
            # reverse order if black
            # adds piece value and position value from table combined if white
            # subtracts piece value if black
            score += (pst[char.upper()][position if char.isupper() else -position] + values.get(char.upper())) * (1 if char.isupper() else -1)
            position += 1
        if char.isdigit():
            position += int(char)

    return score
