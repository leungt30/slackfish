import numpy as np
import chess


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

    piece_order = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]
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


def piece_count(fen_str: str) -> int:
    score_map = {
        "p": -1,
        "n": -3,
        "b": -3,
        "r": -5,
        "q": -9,
        "P": 1,
        "N": 3,
        "B": 3,
        "R": 5,
        "Q": 9,
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
    values = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000}
    pst = {
        "P": (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            78,
            83,
            86,
            73,
            102,
            82,
            85,
            90,
            7,
            29,
            21,
            44,
            40,
            31,
            44,
            7,
            -17,
            16,
            -2,
            15,
            14,
            0,
            15,
            -13,
            -26,
            3,
            10,
            9,
            6,
            1,
            0,
            -23,
            -22,
            9,
            5,
            -11,
            -10,
            -2,
            3,
            -19,
            -31,
            8,
            -7,
            -37,
            -36,
            -14,
            3,
            -31,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        "N": (
            -66,
            -53,
            -75,
            -75,
            -10,
            -55,
            -58,
            -70,
            -3,
            -6,
            100,
            -36,
            4,
            62,
            -4,
            -14,
            10,
            67,
            1,
            74,
            73,
            27,
            62,
            -2,
            24,
            24,
            45,
            37,
            33,
            41,
            25,
            17,
            -1,
            5,
            31,
            21,
            22,
            35,
            2,
            0,
            -18,
            10,
            13,
            22,
            18,
            15,
            11,
            -14,
            -23,
            -15,
            2,
            0,
            2,
            0,
            -23,
            -20,
            -74,
            -23,
            -26,
            -24,
            -19,
            -35,
            -22,
            -69,
        ),
        "B": (
            -59,
            -78,
            -82,
            -76,
            -23,
            -107,
            -37,
            -50,
            -11,
            20,
            35,
            -42,
            -39,
            31,
            2,
            -22,
            -9,
            39,
            -32,
            41,
            52,
            -10,
            28,
            -14,
            25,
            17,
            20,
            34,
            26,
            25,
            15,
            10,
            13,
            10,
            17,
            23,
            17,
            16,
            0,
            7,
            14,
            25,
            24,
            15,
            8,
            25,
            20,
            15,
            19,
            20,
            11,
            6,
            7,
            6,
            20,
            16,
            -7,
            2,
            -15,
            -12,
            -14,
            -15,
            -10,
            -10,
        ),
        "R": (
            35,
            29,
            33,
            4,
            37,
            33,
            56,
            50,
            55,
            29,
            56,
            67,
            55,
            62,
            34,
            60,
            19,
            35,
            28,
            33,
            45,
            27,
            25,
            15,
            0,
            5,
            16,
            13,
            18,
            -4,
            -9,
            -6,
            -28,
            -35,
            -16,
            -21,
            -13,
            -29,
            -46,
            -30,
            -42,
            -28,
            -42,
            -25,
            -25,
            -35,
            -26,
            -46,
            -53,
            -38,
            -31,
            -26,
            -29,
            -43,
            -44,
            -53,
            -30,
            -24,
            -18,
            5,
            -2,
            -18,
            -31,
            -32,
        ),
        "Q": (
            6,
            1,
            -8,
            -104,
            69,
            24,
            88,
            26,
            14,
            32,
            60,
            -10,
            20,
            76,
            57,
            24,
            -2,
            43,
            32,
            60,
            72,
            63,
            43,
            2,
            1,
            -16,
            22,
            17,
            25,
            20,
            -13,
            -6,
            -14,
            -15,
            -2,
            -5,
            -1,
            -10,
            -20,
            -22,
            -30,
            -6,
            -13,
            -11,
            -16,
            -11,
            -16,
            -27,
            -36,
            -18,
            0,
            -19,
            -15,
            -15,
            -21,
            -38,
            -39,
            -30,
            -31,
            -13,
            -31,
            -36,
            -34,
            -42,
        ),
        "K": (
            4,
            54,
            47,
            -99,
            -99,
            60,
            83,
            -62,
            -32,
            10,
            55,
            56,
            56,
            55,
            10,
            3,
            -62,
            12,
            -57,
            44,
            -67,
            28,
            37,
            -31,
            -55,
            50,
            11,
            -4,
            -19,
            13,
            0,
            -49,
            -55,
            -43,
            -52,
            -28,
            -51,
            -47,
            -8,
            -50,
            -47,
            -42,
            -43,
            -79,
            -64,
            -32,
            -29,
            -32,
            -4,
            3,
            -14,
            -50,
            -57,
            -18,
            13,
            4,
            17,
            30,
            -3,
            -14,
            6,
            -1,
            40,
            18,
        ),
    }
    score = 0
    position = 0
    for char in fenstr.split(" ")[0]:
        if char.upper() in pst.keys():
            # uses position to get value from pst table if white
            # reverse order if black
            # adds piece value and position value from table combined if white
            # subtracts piece value if black

            # WIP
            # cannot rotate table if black
            # need to flip vertically, not rotate since board isnt symmetric
            score += (
                pst[char.upper()][position if char.isupper() else -position]
                + values.get(char.upper())
            ) * (1 if char.isupper() else -1)
            position += 1
        if char.isdigit():
            position += int(char)

    return score


def check_one_move_away(fen_str: str) -> int:
    board = chess.Board(fen_str)
    for move in board.legal_moves:
        if board.gives_check(move):
            return 1
    return 0


def legal_moves_per_side(fen_str: str) -> int:
    board = chess.Board(fen_str)
    black_moves = 0
    white_moves = 0
    if board.turn == chess.WHITE:
        white_moves = len(list(board.legal_moves))
    else:
        black_moves = len(list(board.legal_moves))
    board.turn = not board.turn
    if board.turn == chess.WHITE:
        white_moves = len(list(board.legal_moves))
    else:
        black_moves = len(list(board.legal_moves))
    return (black_moves, white_moves)


def issolated_pawns(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_isolated_pawns = np.zeros(64, dtype=np.uint8)
    black_isolated_pawns = np.zeros(64, dtype=np.uint8)
    for square in board.pieces(chess.PAWN, chess.WHITE):
        if is_isolated(board, chess.WHITE, square):
            white_isolated_pawns[square] = 1
    for square in board.pieces(chess.PAWN, chess.BLACK):
        if is_isolated(board, chess.BLACK, square):
            black_isolated_pawns[square] = 1
    return (white_isolated_pawns, black_isolated_pawns)


def is_isolated(board: chess.Board, color: chess.Color, square: int) -> int:
    # check if the pawn square +1 and -1 are not occupied by
    # check the right and left collum if not first and last column
    if square % 8 != 0 and square % 8 != 7:
        # check the right and left collum
        for i in range(0, 63):
            if i % 8 == (square + 1) % 8 or i % 8 == (square - 1) % 8:
                # check if peice here is a pawn
                if (
                    board.piece_at(i) is not None
                    and board.piece_at(i).piece_type == chess.PAWN
                    and board.piece_at(i).color == color
                ):
                    return 0
    else:
        if square % 8 == 0:
            for i in range(0, 63):
                if i % 8 == (square + 1) % 8:
                    # check if peice here is a pawn
                    if (
                        board.piece_at(i) is not None
                        and board.piece_at(i).piece_type == chess.PAWN
                        and board.piece_at(i).color == color
                    ):
                        return 0
        elif square % 8 == 7:
            for i in range(0, 63):
                if i % 8 == (square - 1) % 8:
                    # check if peice here is a pawn
                    if (
                        board.piece_at(i) is not None
                        and board.piece_at(i).piece_type == chess.PAWN
                        and board.piece_at(i).color == color
                    ):
                        return 0
    return 1


def double_pawns(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_double_pawns = np.zeros(64, dtype=np.uint8)
    black_double_pawns = np.zeros(64, dtype=np.uint8)
    for square in board.pieces(chess.PAWN, chess.WHITE):
        # check column, if multiple in same colmn add to bitboard
        for i in range(0, 63):
            if i != square and i % 8 == square % 8:  # same column
                if (
                    board.piece_at(i) is not None
                    and board.piece_at(i).piece_type == chess.PAWN
                    and board.piece_at(i).color == chess.WHITE
                ):
                    white_double_pawns[square] = 1
                    white_double_pawns[i] = 1
    for square in board.pieces(chess.PAWN, chess.BLACK):
        for i in range(0, 63):
            if i != square and i % 8 == square % 8:  # same column
                if (
                    board.piece_at(i) is not None
                    and board.piece_at(i).piece_type == chess.PAWN
                    and board.piece_at(i).color == chess.BLACK
                ):
                    black_double_pawns[square] = 1
                    black_double_pawns[i] = 1
    return (white_double_pawns, black_double_pawns)


def rook_on_7th_rank(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_rook_on_7th_rank = np.zeros(64, dtype=np.uint8)
    black_rook_on_7th_rank = np.zeros(64, dtype=np.uint8)
    for square in board.pieces(chess.ROOK, chess.WHITE):
        if square // 8 == 6:  # 7th rank for white
            white_rook_on_7th_rank[square] = 1
    for square in board.pieces(chess.ROOK, chess.BLACK):
        if square // 8 == 1:  # 2nd rank for black
            black_rook_on_7th_rank[square] = 1
    return (white_rook_on_7th_rank, black_rook_on_7th_rank)


def rook_on_semi_open_file(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_rook_on_semi_open_file = np.zeros(64, dtype=np.uint8)
    black_rook_on_semi_open_file = np.zeros(64, dtype=np.uint8)
    for square in board.pieces(chess.ROOK, chess.WHITE):
        if is_semi_open_file(board, chess.WHITE, square):
            white_rook_on_semi_open_file[square] = 1
    for square in board.pieces(chess.ROOK, chess.BLACK):
        if is_semi_open_file(board, chess.BLACK, square):
            black_rook_on_semi_open_file[square] = 1
    return (white_rook_on_semi_open_file, black_rook_on_semi_open_file)


def is_semi_open_file(board: chess.Board, color: chess.Color, square: int) -> int:
    for i in range(0, 63):
        if i != square and i % 8 == square % 8:
            if (
                board.piece_at(i) is not None
                and board.piece_at(i).piece_type == chess.PAWN
                and board.piece_at(i).color == color
            ):
                return 0
    return 1


def rooks_on_same_file(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_rooks_on_same_file = np.zeros(64, dtype=np.uint8)
    black_rooks_on_same_file = np.zeros(64, dtype=np.uint8)
    for square in board.pieces(chess.ROOK, chess.WHITE):
        # check column, if multiple in same colmn add to bitboard
        for i in range(0, 63):
            if i != square and i % 8 == square % 8:  # same column
                if (
                    board.piece_at(i) is not None
                    and board.piece_at(i).piece_type == chess.ROOK
                    and board.piece_at(i).color == chess.WHITE
                ):
                    white_rooks_on_same_file[square] = 1
                    white_rooks_on_same_file[i] = 1
    for square in board.pieces(chess.ROOK, chess.BLACK):
        for i in range(0, 63):
            if i != square and i % 8 == square % 8:  # same column
                if (
                    board.piece_at(i) is not None
                    and board.piece_at(i).piece_type == chess.ROOK
                    and board.piece_at(i).color == chess.BLACK
                ):
                    black_rooks_on_same_file[square] = 1
                    black_rooks_on_same_file[i] = 1
    return (white_rooks_on_same_file, black_rooks_on_same_file)


def is_forking(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_forking = 0
    black_forking = 0

    # Check white knights
    for knight_square in board.pieces(chess.KNIGHT, chess.WHITE):
        attacked_pieces = 0
        # Get all squares this specific knight attacks
        attacks = board.attacks(knight_square)
        for target_square in attacks:
            # Check if there's an enemy piece on this square
            piece = board.piece_at(target_square)
            if piece is not None and piece.color == chess.BLACK:
                attacked_pieces += 1
        if attacked_pieces > 1:
            white_forking += 1

    # Check black knights
    for knight_square in board.pieces(chess.KNIGHT, chess.BLACK):
        attacked_pieces = 0
        # Get all squares this specific knight attacks
        attacks = board.attacks(knight_square)
        for target_square in attacks:
            # Check if there's an enemy piece on this square
            piece = board.piece_at(target_square)
            if piece is not None and piece.color == chess.WHITE:
                attacked_pieces += 1
        if attacked_pieces > 1:
            black_forking += 1

    return (white_forking, black_forking)


def fork_available(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_fork_available = 0
    black_fork_available = 0
    # make a knight move, use is_forking to check if it is a fork for that color
    for knight_square in board.pieces(chess.KNIGHT, chess.WHITE):
        attacks = board.attacks(knight_square)
        for target_square in attacks:
            # play the move
            board.push(chess.Move(knight_square, target_square))
            if is_forking(board.fen())[0] == 1:
                white_fork_available += 1
            board.pop()
    for knight_square in board.pieces(chess.KNIGHT, chess.BLACK):
        attacks = board.attacks(knight_square)
        for target_square in attacks:
            board.push(chess.Move(knight_square, target_square))
            if is_forking(board.fen())[1] == 1:
                black_fork_available += 1
            board.pop()
    return (white_fork_available, black_fork_available)


def bishop_activity(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_bishop_activity = 0
    black_bishop_activity = 0
    for square in board.pieces(chess.BISHOP, chess.WHITE):
        attacks = board.attacks(square)
        for target_square in attacks:
            if (
                board.piece_at(target_square) is None
                or board.piece_at(target_square).color == chess.BLACK
            ):
                white_bishop_activity += 1
    for square in board.pieces(chess.BISHOP, chess.BLACK):
        attacks = board.attacks(square)
        for target_square in attacks:
            if (
                board.piece_at(target_square) is None
                or board.piece_at(target_square).color == chess.WHITE
            ):
                black_bishop_activity += 1
    return (white_bishop_activity, black_bishop_activity)


def pinned_pieces(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_pinned_pieces = 0
    black_pinned_pieces = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                if board.is_pinned(chess.WHITE, square):
                    white_pinned_pieces += 1
            else:
                if board.is_pinned(chess.BLACK, square):
                    black_pinned_pieces += 1
    return (white_pinned_pieces, black_pinned_pieces)


def number_of_hanging_pieces(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_hanging_pieces = 0
    black_hanging_pieces = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                if board.is_attacked_by(
                    chess.BLACK, square
                ) and not board.is_attacked_by(chess.WHITE, square):
                    white_hanging_pieces += 1
            else:
                if board.is_attacked_by(
                    chess.WHITE, square
                ) and not board.is_attacked_by(chess.BLACK, square):
                    black_hanging_pieces += 1
    return (white_hanging_pieces, black_hanging_pieces)


def hanging_pieces_bitboards(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_hanging_pieces = np.zeros(64, dtype=np.uint8)
    black_hanging_pieces = np.zeros(64, dtype=np.uint8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                if board.is_attacked_by(
                    chess.BLACK, square
                ) and not board.is_attacked_by(chess.WHITE, square):
                    white_hanging_pieces[square] = 1
            else:
                if board.is_attacked_by(
                    chess.WHITE, square
                ) and not board.is_attacked_by(chess.BLACK, square):
                    black_hanging_pieces[square] = 1
    return (white_hanging_pieces, black_hanging_pieces)


def center_controll(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_center_control = 0
    black_center_control = 0
    center_squares = ["e4", "e5", "d4", "d5"]
    for square in chess.SQUARES:
        if chess.square_name(square) in center_squares:
            white_center_control += len(list(board.attackers(chess.WHITE, square)))
            black_center_control += len(list(board.attackers(chess.BLACK, square)))
    return (white_center_control, black_center_control)
